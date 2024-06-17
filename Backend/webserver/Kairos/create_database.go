package Kairos

import (
	"log"

	"gorm.io/driver/postgres"
	"gorm.io/gorm"
)

var DB *gorm.DB

// 定义表结构
type AberrationStaticsTable struct {
	TimeInterval string  `gorm:"column:time_interval"`
	LossAvg      float64 `gorm:"column:loss_avg"`
	Count        int64   `gorm:"column:count"`
	Percentage   float64 `gorm:"column:percentage"`
	NodeNum      int64   `gorm:"column:node_num"`
	EdgeNum      int64   `gorm:"column:edge_num"`
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
	Time        int64  `gorm:"column:time"`
	SubjectType string `gorm:"column:subject_type"`
	SubjectName string `gorm:"column:subject_name"`
}

type AnomalousSubject struct {
	Time        int64  `gorm:"column:time"`
	SubjectType string `gorm:"column:subject_type"`
	SubjectName string `gorm:"column:subject_name"`
}

type DangerousAction struct {
	Time        int64  `gorm:"column:time"`
	SubjectType string `gorm:"column:subject_type"`
	SubjectName string `gorm:"column:subject_name"`
	Action      string `gorm:"column:action"`
	ObjectType  string `gorm:"column:object_type"`
	ObjectName  string `gorm:"column:object_name"`
}

type AnomalousAction struct {
	Time        int64  `gorm:"column:time"`
	SubjectType string `gorm:"column:subject_type"`
	SubjectName string `gorm:"column:subject_name"`
	Action      string `gorm:"column:action"`
	ObjectType  string `gorm:"column:object_type"`
	ObjectName  string `gorm:"column:object_name"`
}

type DangerousObject struct {
	Time       int64  `gorm:"column:time"`
	ObjectType string `gorm:"column:object_type"`
	ObjectName string `gorm:"column:object_name"`
}

type AnomalousObject struct {
	Time       int64  `gorm:"column:time"`
	ObjectType string `gorm:"column:object_type"`
	ObjectName string `gorm:"column:object_name"`
}
