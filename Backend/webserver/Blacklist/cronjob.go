package Blacklist

import (
	"encoding/json"
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
	DB.Where("timestamp_rec >= ? AND timestamp_rec <= ?", startTime, endTime).Find(&actions)
	for _, action := range actions {
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
