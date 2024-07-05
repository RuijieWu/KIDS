package Blacklist

import (
	"log"

	"gorm.io/driver/postgres"
	"gorm.io/gorm"
)

var DB *gorm.DB

// 表结构定义
type DangerousSubject struct {
	gorm.Model
	Exec string `gorm:"column:exec"`
}

func (DangerousSubject) TableName() string {
	return "dangerous_subjects_table"
}

type DangerousAction struct {
	Time        int64  `gorm:"column:time"`
	SubjectType string `gorm:"column:subject_type"`
	SubjectName string `gorm:"column:subject_name"`
	Action      string `gorm:"column:action"`
	ObjectType  string `gorm:"column:object_type"`
	ObjectName  string `gorm:"column:object_name"`
}

func (DangerousAction) TableName() string {
	return "dangerous_actions_table"
}

type DangerousFile struct {
	Path string `gorm:"column:path"`
}

func (DangerousFile) TableName() string {
	return "dangerous_files_table"
}

type DangerousNetFlow struct {
	gorm.Model
	LocalAddr  string `gorm:"column:src_addr"`
	LocalPort  string `gorm:"column:src_port"`
	RemoteAddr string `gorm:"column:dst_addr"`
	RemotePort string `gorm:"column:dst_port"`
}

func (DangerousNetFlow) TableName() string {
	return "dangerous_net_flows_table"
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
		&DangerousSubject{},
		&DangerousAction{},
		&DangerousFile{},
		&DangerousNetFlow{},
	); err != nil {
		log.Printf("Failed to migrate tables: %v", err)
		log.Fatal("failed to migrate database")
	}

	log.Println("Database initialized successfully")
}
