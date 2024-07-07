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
	// Set the time range for fetching the audit logs
	startTime := time.Now().Add(-10 * time.Second).Format("2006-01-02T15:04:05")
	endTime := time.Now().Format("2006-01-02T15:04:05")

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
	InsertBlacklistActions(startTime, endTime)
}

// check the events in past 10 seconds and insert the blacklisted actions. set the flag to 0
func InsertBlacklistActions(startTime string, endTime string) {
	var actions []audit_data.Event

	// convert the time string to time.Time
	startTimeParsed, err := time.Parse("2006-01-02T15:04:05", startTime)
	if err != nil {
		log.Printf("Failed to parse start time: %v", err)
		return
	}
	endTimeParsed, err := time.Parse("2006-01-02T15:04:05", endTime)
	if err != nil {
		log.Printf("Failed to parse end time: %v", err)
		return
	}

	// 将 time.Time 对象转换为 Unix 时间戳
	startTimeUnix := startTimeParsed.Unix()
	endTimeUnix := endTimeParsed.Unix()

	DB.Where("timestamp_rec >= ? AND timestamp_rec <= ?", startTimeUnix, endTimeUnix).Find(&actions)
	for _, action := range actions {
		// if the action's subject or object is in the blacklist, insert the action
		if IsBlacklisted(action.SrcNode) || IsBlacklisted(action.DstNode) {
			blacklistAction := BlacklistAction{
				SrcNode:      action.SrcNode,
				SrcIndexID:   action.SrcIndexID,
				Operation:    action.Operation,
				DstNode:      action.DstNode,
				DstIndexID:   action.DstIndexID,
				TimestampRec: action.TimestampRec,
				Flag:         0,
			}
			// compare all parms except flag; if the action does not exist, insert it
			DB.Where("src_node = ? AND src_index_id = ? AND operation = ? AND dst_node = ? AND dst_index_id = ? AND timestamp_rec = ?",
				blacklistAction.SrcNode, blacklistAction.SrcIndexID, blacklistAction.Operation, blacklistAction.DstNode, blacklistAction.DstIndexID, blacklistAction.TimestampRec).FirstOrCreate(&blacklistAction)
		}
	}
}

// check if the node is in the blacklist
// first get the uuid and search for it in audit_data's nodes; then search for the node in the blacklist
func IsBlacklisted(node string) bool {
	// search for the node in the audit_data's node2uuid
	var node2uuid audit_data.NodeID
	fmt.Println(node)
	DB.Where("node = ?", node).First(&node2uuid)

	if node2uuid.Type == "netflow" {
		var orignNode audit_data.NetFlowNode
		DB.Where("hash_id = ?", node2uuid.Hash).First(&orignNode)
		var blackNode BlacklistNetFlow
		DB.Where("src_addr = ? AND src_port = ? AND dst_addr = ? AND dst_port = ?", orignNode.LocalAddr, orignNode.LocalPort, orignNode.RemoteAddr, orignNode.RemotePort).First(&blackNode)
		if blackNode.ID != 0 {
			return true
		} else {
			return false
		}
	}

	if node2uuid.Type == "subject" {
		var orignNode audit_data.SubjectNode
		DB.Where("hash_id = ?", node2uuid.Hash).First(&orignNode)
		var blackNode BlacklistSubject
		DB.Where("exec = ?", orignNode.Exec).First(&blackNode)
		if blackNode.ID != 0 {
			return true
		} else {
			return false
		}
	}

	if node2uuid.Type == "file" {
		var orignNode audit_data.FileNode
		DB.Where("hash_id = ?", node2uuid.Hash).First(&orignNode)
		var blackNode BlacklistFile
		DB.Where("path = ?", orignNode.Path).First(&blackNode)
		if blackNode.ID != 0 {
			return true
		} else {
			return false
		}
	}

	return false
}
