package Kairos

import (
	"log"
	"time"

	"gorm.io/driver/postgres"
	"gorm.io/gorm"
	"gorm.io/gorm/logger"
)

var DB *gorm.DB

// 定义表结构
type AberrationStaticsTable struct {
	BeginTime  int64   `gorm:"column:begin_time"`
	EndTime    int64   `gorm:"column:end_time"`
	LossAvg    float64 `gorm:"column:loss_avg"`
	Count      int64   `gorm:"column:count"`
	Percentage float64 `gorm:"column:percentage"`
	NodeNum    int64   `gorm:"column:node_num"`
	EdgeNum    int64   `gorm:"column:edge_num"`
}

func (AberrationStaticsTable) TableName() string {
	return "aberration_statics_table"
}

// 初始化数据库连接并创建表
func InitKairosDatabase() {
	dsn := "host=/var/run/postgresql/ user=postgres password=postgres dbname=tc_cadet_dataset_db port=5432 sslmode=disable TimeZone=Asia/Shanghai client_encoding=UTF8"
	var err error
	DB, err = gorm.Open(postgres.Open(dsn), &gorm.Config{
		Logger: logger.Default.LogMode(logger.Silent),
	})
	if err != nil {
		log.Printf("Failed to connect to database: %v", err)
		log.Fatal("failed to connect database")
	}

	// 自动迁移创建所有表
	if err := DB.AutoMigrate(
		&AberrationStaticsTable{},
		&DangerousSubject{},
		&AnomalousSubject{},
		&DangerousAction{},
		&AnomalousAction{},
		&DangerousObject{},
		&AnomalousObject{},
	); err != nil {
		log.Printf("Failed to migrate tables: %v", err)
		log.Fatal("failed to migrate database")
	}

	log.Println("Database initialized successfully")
}

// 表结构定义
type DangerousSubject struct {
	Timestamp   time.Time `gorm:"column:timestamp"`
	Time        int64     `gorm:"column:time"`
	SubjectType string    `gorm:"column:subject_type"`
	SubjectName string    `gorm:"column:subject_name"`
	GraphIndex  string    `gorm:"column:graph_index"`
}

func (DangerousSubject) TableName() string {
	return "dangerous_subjects_table"
}

type AnomalousSubject struct {
	Timestamp   time.Time `gorm:"column:timestamp"`
	Time        int64     `gorm:"column:time"`
	SubjectType string    `gorm:"column:subject_type"`
	SubjectName string    `gorm:"column:subject_name"`
	GraphIndex  string    `gorm:"column:graph_index"`
}

func (AnomalousSubject) TableName() string {
	return "anomalous_subjects_table"
}

type DangerousAction struct {
	Timestamp   time.Time `gorm:"column:timestamp"`
	Time        int64     `gorm:"column:time"`
	SubjectType string    `gorm:"column:subject_type"`
	SubjectName string    `gorm:"column:subject_name"`
	Action      string    `gorm:"column:action"`
	ObjectType  string    `gorm:"column:object_type"`
	ObjectName  string    `gorm:"column:object_name"`
	GraphIndex  string    `gorm:"column:graph_index"`
}

func (DangerousAction) TableName() string {
	return "dangerous_actions_table"
}

type AnomalousAction struct {
	Timestamp   time.Time `gorm:"column:timestamp"`
	Time        int64     `gorm:"column:time"`
	SubjectType string    `gorm:"column:subject_type"`
	SubjectName string    `gorm:"column:subject_name"`
	Action      string    `gorm:"column:action"`
	ObjectType  string    `gorm:"column:object_type"`
	ObjectName  string    `gorm:"column:object_name"`
	GraphIndex  string    `gorm:"column:graph_index"`
}

func (AnomalousAction) TableName() string {
	return "anomalous_actions_table"
}

type DangerousObject struct {
	Timestamp  time.Time `gorm:"column:timestamp"`
	Time       int64     `gorm:"column:time"`
	ObjectType string    `gorm:"column:object_type"`
	ObjectName string    `gorm:"column:object_name"`
	GraphIndex string    `gorm:"column:graph_index"`
}

func (DangerousObject) TableName() string {
	return "dangerous_objects_table"
}

type AnomalousObject struct {
	Timestamp  time.Time `gorm:"column:timestamp"`
	Time       int64     `gorm:"column:time"`
	ObjectType string    `gorm:"column:object_type"`
	ObjectName string    `gorm:"column:object_name"`
	GraphIndex string    `gorm:"column:graph_index"`
}

func (AnomalousObject) TableName() string {
	return "anomalous_objects_table"
}
