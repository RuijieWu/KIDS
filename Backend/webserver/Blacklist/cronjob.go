package Blacklist

import (
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"strconv"
	"time"

	"gorm.io/driver/postgres"
	"gorm.io/gorm"
	"gorm.io/gorm/logger"

	"KIDS/audit_data"
)

func init() {
	var err error
	dsn := "host=/var/run/postgresql/ user=postgres password=postgres dbname=tc_cadet_dataset_db port=5432 sslmode=disable TimeZone=Asia/Shanghai"
	DB, err = gorm.Open(postgres.Open(dsn), &gorm.Config{
		Logger: logger.Default.LogMode(logger.Silent),
	})
	if err != nil {
		log.Fatalf("failed to connect to database: %v", err)
	}
}

func Cronjob() {
	ticker := time.NewTicker(15	 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-ticker.C:
			processAuditLogs()
		}
	}
}

func processAuditLogs() {
	location, err := time.LoadLocation("Asia/Shanghai")
	if err != nil {
		fmt.Println("Error loading location:", err)
		return
	}

	// 获取当前的时间并转换为 Asia/Shanghai 时区
	now := time.Now().In(location)
	startTime := now.Add(-10 * time.Second).Format("2006-01-02T15:04:05")
	startTimeUnix := now.Add(-10 * time.Second).UnixNano()
	endTime := now.Format("2006-01-02T15:04:05")
	endTimeUnix := now.UnixNano()

	pythonURLs := audit_data.AgentIPs

	for _, pythonURL := range pythonURLs {
		resp, err := http.Get("http://" + pythonURL + "/audit-logs" + "?start_time=" + startTime + "&end_time=" + endTime)
		if err != nil {
			// log.Printf("Error getting audit logs: %v", err)
			continue
		}
		defer resp.Body.Close()

		body, err := io.ReadAll(resp.Body)
		if err != nil {
			log.Printf("Error reading response body: %v", err)
			continue
		}

		var result map[string]interface{}
		if err := json.Unmarshal(body, &result); err != nil {
			log.Printf("Error unmarshalling response body: %v", err)
			continue
		}

		// Parse to type Events struct
		var events audit_data.Events
		if err := json.Unmarshal(body, &events); err != nil {
			continue
		}

		// print the events
		log.Printf("Events: %v", events)

		// Filter and insert events
		// if in white list, delete the event
		newEvents := audit_data.Events{}
		for _, folder := range events.FolderWatch {
			if _, ok := IsWhitelisted(folder); ok {
				continue
			} else {
				newEvents.FolderWatch = append(newEvents.FolderWatch, folder)
			}
		}

		for _, netflow := range events.SocketOps {
			if _, ok := IsWhitelisted(netflow); ok {
				continue
			} else {
				newEvents.SocketOps = append(newEvents.SocketOps, netflow)
			}
		}

		events = newEvents

		audit_data.InsertEvents(events)
		InsertBlacklistActions(fmt.Sprintf("%d", startTimeUnix), fmt.Sprintf("%d", endTimeUnix))
	}
}

// check the events in past 10 seconds and insert the blacklisted actions. set the flag to 0
func InsertBlacklistActions(startTimeUnix string, endTimeUnix string) {
	var actions []audit_data.Event

	DB.Where("timestamp_rec >= ? AND timestamp_rec <= ?", startTimeUnix, endTimeUnix).Find(&actions)
	for _, action := range actions {
		// if the action's subject or object is in the blacklist, insert the action
		if subject, subjectType, ok := IsBlacklisted(action.SrcNode); ok {
			DB.Create(&BlacklistAction{
				TargetName:   subject,
				TargetType:   subjectType,
				TimestampRec: action.TimestampRec,
				Flag:         0,
			})
		}
		if object, objectType, ok := IsBlacklisted(action.DstNode); ok {
			DB.Create(&BlacklistAction{
				TargetName:   object,
				TargetType:   objectType,
				TimestampRec: action.TimestampRec,
				Flag:         0,
			})
		}
	}
}

// check if the node is in the blacklist
// first get the uuid and search for it in audit_data's nodes; then search for the node in the blacklist
func IsBlacklisted(node string) (string, string, bool) {
	// search for the node in the audit_data's node2uuid
	var node2uuid audit_data.NodeID
	DB.Where("hash_id = ?", node).First(&node2uuid)

	if node2uuid.Type == "netflow" {
		var orignNode audit_data.NetFlowNode
		DB.Where("hash_id = ?", node2uuid.Hash).First(&orignNode)
		var blackNode BlacklistNetFlow
		DB.Where("src_addr = ? AND src_port = ? AND dst_addr = ? AND dst_port = ?", orignNode.LocalAddr, orignNode.LocalPort, orignNode.RemoteAddr, orignNode.RemotePort).First(&blackNode)
		if blackNode.ID != 0 {
			return fmt.Sprintf("%s:%s -> %s:%s", orignNode.LocalAddr, orignNode.LocalPort, orignNode.RemoteAddr, orignNode.RemotePort), "netflow", true
		} else {
			return "", "", false
		}
	}

	if node2uuid.Type == "subject" {
		var orignNode audit_data.SubjectNode
		DB.Where("hash_id = ?", node2uuid.Hash).First(&orignNode)
		var blackNode BlacklistSubject
		DB.Where("exec = ?", orignNode.Exec).First(&blackNode)
		if blackNode.ID != 0 {
			return orignNode.Exec, "subject", true
		} else {
			return "", "", false
		}
	}

	if node2uuid.Type == "file" {
		var orignNode audit_data.FileNode
		DB.Where("hash_id = ?", node2uuid.Hash).First(&orignNode)
		var blackNode BlacklistFile
		DB.Where("path = ?", orignNode.Path).First(&blackNode)
		if blackNode.ID != 0 {
			return orignNode.Path, "file", true
		} else {
			return "", "", false
		}
	}

	return "", "", false
}

func IsWhitelisted(node interface{}) (string, bool) {
	switch node := node.(type) {
	case audit_data.FolderWatch:
		var whiteNode WhitelistFile
		DB.Where("path = ?", node.File).First(&whiteNode)
		if whiteNode.ID != 0 {
			return node.File, true
		}
		var whiteNode2 WhitelistSubject
		DB.Where("exec = ?", node.Process).First(&whiteNode2)
		if whiteNode2.ID != 0 {
			return node.Process, true
		}
		return "", false

	case audit_data.SocketOperation:
		var whiteNode WhitelistSubject
		DB.Where("exec = ?", node.Process).First(&whiteNode)
		if whiteNode.ID != 0 {
			return node.Process, true
		}
		var whiteNode2 WhitelistNetFlow
		DB.Where("src_addr = ? AND src_port = ? AND dst_addr = ? AND dst_port = ?", node.SourceIP, node.SourcePort, node.DestinationIP, node.DestinationPort).First(&whiteNode2)
		if whiteNode2.ID != 0 {
			return fmt.Sprintf("%s:%s -> %s:%s", node.SourceIP, strconv.Itoa(node.SourcePort), node.DestinationIP, strconv.Itoa(node.DestinationPort)), true
		}
		return "", false

	default:
		return "", false
	}
}
