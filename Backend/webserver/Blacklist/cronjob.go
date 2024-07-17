package Blacklist

import (
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"time"

	"gorm.io/driver/postgres"
	"gorm.io/gorm"

	"KIDS/audit_data"
)

func init() {
	var err error
	dsn := "host=/var/run/postgresql/ user=postgres password=postgres dbname=tc_cadet_dataset_db port=5432 sslmode=disable TimeZone=Asia/Shanghai"
	DB, err = gorm.Open(postgres.Open(dsn), &gorm.Config{})
	if err != nil {
		log.Fatalf("failed to connect to database: %v", err)
	}
}

func Cronjob() {
	ticker := time.NewTicker(10 * time.Second)
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

	pythonURL := "http://localhost:8010/audit-logs"
	resp, err := http.Get(pythonURL + "?start_time=" + startTime + "&end_time=" + endTime)
	if err != nil {
		log.Printf("Failed to connect to the Python service: %v", err)
		return
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		log.Printf("Failed to read response from the Python service: %v", err)
		return
	}

	var result map[string]interface{}
	if err := json.Unmarshal(body, &result); err != nil {
		log.Printf("Failed to parse response from the Python service: %v", err)
		return
	}

	// Parse to type Events struct
	var events audit_data.Events
	if err := json.Unmarshal(body, &events); err != nil {
		log.Printf("Failed to parse events from the response: %v", err)
		return
	}

	// Filter and insert events
	audit_data.InsertEvents(events)
	InsertBlacklistActions(fmt.Sprintf("%d", startTimeUnix), fmt.Sprintf("%d", endTimeUnix))
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
