package Blacklist

import (
	"context"
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

	// 初始化 Hive 连接
	InitHiveConnection()
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

	var events audit_data.Events
	if err := json.Unmarshal(body, &events); err != nil {
		log.Printf("Failed to parse response from the Python service: %v", err)
		return
	}

	// 插入事件到 PostgreSQL
	audit_data.InsertEvents(events)

	// 将数据迁移到 Hive
	migrateDataToHive(fmt.Sprintf("%d", startTimeUnix), fmt.Sprintf("%d", endTimeUnix))

	// 插入黑名单动作到 Hive 和 PostgreSQL
	InsertBlacklistActions(fmt.Sprintf("%d", startTimeUnix), fmt.Sprintf("%d", endTimeUnix))
}

func migrateDataToHive(startTimeUnix string, endTimeUnix string) {
	ctx := context.Background()
	cursor := hiveConn.Cursor()

	var actions []audit_data.Event
	DB.Where("timestamp_rec >= ? AND timestamp_rec <= ?", startTimeUnix, endTimeUnix).Find(&actions)

	for _, action := range actions {
		query := fmt.Sprintf("INSERT INTO audit_data_event (src_node, dst_node, timestamp_rec) VALUES ('%s', '%s', %d)", action.SrcNode, action.DstNode, action.TimestampRec)
		cursor.Exec(ctx, query)
		if cursor.Err != nil {
			log.Printf("Failed to insert into audit_data_event: %v", cursor.Err)
		}
	}
}

func InsertBlacklistActions(startTimeUnix string, endTimeUnix string) {
	migrateDataToHive(startTimeUnix, endTimeUnix)

	ctx := context.Background()
	cursor := hiveConn.Cursor()

	query := `
		SELECT e.src_node, e.dst_node, e.timestamp_rec
		FROM audit_data_event e
		LEFT JOIN blacklist_netflows_table bnet ON e.src_node = bnet.src_addr
		LEFT JOIN blacklist_subjects_table bsub ON e.src_node = bsub.exec
		LEFT JOIN blacklist_files_table bfile ON e.src_node = bfile.path
		WHERE bnet.src_addr IS NOT NULL OR bsub.exec IS NOT NULL OR bfile.path IS NOT NULL
		UNION
		SELECT e.src_node, e.dst_node, e.timestamp_rec
		FROM audit_data_event e
		LEFT JOIN blacklist_netflows_table bnet ON e.dst_node = bnet.src_addr
		LEFT JOIN blacklist_subjects_table bsub ON e.dst_node = bsub.exec
		LEFT JOIN blacklist_files_table bfile ON e.dst_node = bfile.path
		WHERE bnet.src_addr IS NOT NULL OR bsub.exec IS NOT NULL OR bfile.path IS NOT NULL
	`

	// 执行查询
	cursor.Exec(ctx, query)
	if cursor.Err != nil {
		log.Fatalf("Failed to query Hive: %v", cursor.Err)
	}
	defer cursor.Close()

	var srcNode, dstNode string
	var timestampRec int64

	for cursor.HasMore(ctx) {
		if cursor.Err != nil {
			log.Printf("Failed to iterate over rows: %v", cursor.Err)
			continue
		}

		cursor.FetchOne(ctx, &srcNode, &dstNode, &timestampRec)
		if cursor.Err != nil {
			log.Printf("Failed to fetch row: %v", cursor.Err)
			continue
		}

		// 在 PostgreSQL 中插入结果
		if srcNode != "" {
			err := DB.Create(&BlacklistAction{
				TargetName:   srcNode,
				TargetType:   "src_node",
				TimestampRec: timestampRec,
				Flag:         0,
			}).Error
			if err != nil {
				log.Printf("Failed to insert into PostgreSQL: %v", err)
			}
			query := fmt.Sprintf("INSERT INTO blacklist_actions_table (target_name, target_type, timestamp_rec, flag) VALUES ('%s', 'src_node', %d, 0)", srcNode, timestampRec)
			cursor.Exec(ctx, query)
			if cursor.Err != nil {
				log.Printf("Failed to insert into Hive: %v", cursor.Err)
			}
		}
		if dstNode != "" {
			err := DB.Create(&BlacklistAction{
				TargetName:   dstNode,
				TargetType:   "dst_node",
				TimestampRec: timestampRec,
				Flag:         0,
			}).Error
			if err != nil {
				log.Printf("Failed to insert into PostgreSQL: %v", err)
			}
			query := fmt.Sprintf("INSERT INTO blacklist_actions_table (target_name, target_type, timestamp_rec, flag) VALUES ('%s', 'dst_node', %d, 0)", dstNode, timestampRec)
			cursor.Exec(ctx, query)
			if cursor.Err != nil {
				log.Printf("Failed to insert into Hive: %v", cursor.Err)
			}
		}
	}

	cursor.Close()
	hiveConn.Close()
}
