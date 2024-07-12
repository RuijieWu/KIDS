package Blacklist

import (
	"log"

	"gorm.io/driver/postgres"
	"gorm.io/gorm"
	"github.com/beltran/gohive"
)

var DB *gorm.DB

type BlacklistAction struct {
	gorm.Model
	TargetName   string `gorm:"column:target_name"`
	TargetType   string `gorm:"column:target_type"`
	TimestampRec int64  `gorm:"column:timestamp_rec"`
	Flag         int    `gorm:"column:flag"`
}

func (BlacklistAction) TableName() string {
	return "blacklist_actions_table"
}

// 表结构定义
type BlacklistSubject struct {
	gorm.Model
	Exec string `gorm:"column:exec;uniqueIndex:unique_blacklist_subjects"`
}

func (BlacklistSubject) TableName() string {
	return "blacklist_subjects_table"
}

type BlacklistFile struct {
	gorm.Model
	Path string `gorm:"column:path;uniqueIndex:unique_blacklist_files"`
}

func (BlacklistFile) TableName() string {
	return "blacklist_files_table"
}

type BlacklistNetFlow struct {
	gorm.Model
	LocalAddr  string `gorm:"column:src_addr;uniqueIndex:unique_blacklist_net_flows" json:"src_addr"`
	LocalPort  string `gorm:"column:src_port;uniqueIndex:unique_blacklist_net_flows" json:"src_port"`
	RemoteAddr string `gorm:"column:dst_addr;uniqueIndex:unique_blacklist_net_flows" json:"dst_addr"`
	RemotePort string `gorm:"column:dst_port;uniqueIndex:unique_blacklist_net_flows" json:"dst_port"`
}

func (BlacklistNetFlow) TableName() string {
	return "blacklist_netflows_table"
}

// 初始化数据库连接并创建表
func InitBlacklistDatabase() {
	dsn := "host=/var/run/postgresql/ user=postgres password=postgres dbname=tc_cadet_dataset_db port=5432 sslmode=disable TimeZone=Asia/Shanghai client_encoding=UTF8"
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

func InitHiveConnection() {
	configuration := gohive.NewConnectConfiguration()
	// 设置 Hive 连接参数，根据实际情况填写
	configuration.Username = "your_hive_username"
	configuration.Password = "your_hive_password"

	var err error
	hiveConn, err = gohive.Connect("your_hive_host", 10000, "NOSASL", configuration)
	if err != nil {
		log.Fatalf("Failed to connect to Hive: %v", err)
	}
	log.Println("Connected to Hive successfully")
}