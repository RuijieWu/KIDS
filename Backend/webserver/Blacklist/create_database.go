package Blacklist

import (
	"log"

	"gorm.io/driver/postgres"
	"gorm.io/gorm"
)

var DB *gorm.DB

// 表结构定义
type BlacklistSubject struct {
	gorm.Model
	Exec string `gorm:"column:exec"`
}

func (BlacklistSubject) TableName() string {
	return "blacklist_subjects_table"
}

type BlacklistAction struct {
	gorm.Model
	SrcNode      string `gorm:"column:src_node"`
	SrcIndexID   string `gorm:"column:src_index_id"`
	Operation    string `gorm:"column:operation"`
	DstNode      string `gorm:"column:dst_node"`
	DstIndexID   string `gorm:"column:dst_index_id"`
	TimestampRec int64  `gorm:"column:timestamp_rec"`
	Flag 	   	 int    `gorm:"column:flag"`
}

func (BlacklistAction) TableName() string {
	return "blacklist_actions_table"
}

type BlacklistFile struct {
	Path string `gorm:"column:path"`
}

func (BlacklistFile) TableName() string {
	return "blacklist_files_table"
}

type BlacklistNetFlow struct {
	gorm.Model
	LocalAddr  string `gorm:"column:src_addr"`
	LocalPort  string `gorm:"column:src_port"`
	RemoteAddr string `gorm:"column:dst_addr"`
	RemotePort string `gorm:"column:dst_port"`
}

func (BlacklistNetFlow) TableName() string {
	return "Blacklist_net_flows_table"
}

// 初始化数据库连接并创建表
func InitKairosDatabase() {
	dsn := "host=/var/run/postgresql/ user=postgres password=postgres dbname=tc_cadet_dataset_db port=5432 sslmode=disable TimeZone=Asia/Shanghai"
	var err error
	DB, err = gorm.Open(postgres.Open(dsn), &gorm.Config{})
	if err != nil {
		log.Printf("Failed to connect to database: %v", err)
		log.Fatal("failed to connect database")
	}

	// 自动迁移创建所有表
	if err := DB.AutoMigrate(
		&BlacklistSubject{},
		&BlacklistAction{},
		&BlacklistFile{},
		&BlacklistNetFlow{},
	); err != nil {
		log.Printf("Failed to migrate tables: %v", err)
		log.Fatal("failed to migrate database")
	}

	log.Println("Database initialized successfully")
}
