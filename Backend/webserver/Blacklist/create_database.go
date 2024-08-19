package Blacklist

import (
	"log"

	"gorm.io/driver/postgres"
	"gorm.io/gorm"
	"gorm.io/gorm/logger"
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

type WhitelistSubject struct {
	gorm.Model
	Exec string `gorm:"column:exec;uniqueIndex:unique_whitelist_subjects"`
}

func (WhitelistSubject) TableName() string {
	return "whitelist_subjects_table"
}

type WhitelistFile struct {
	gorm.Model
	Path string `gorm:"column:path;uniqueIndex:unique_whitelist_files"`
}

func (WhitelistFile) TableName() string {
	return "whitelist_files_table"
}

type WhitelistNetFlow struct {
	gorm.Model
	LocalAddr  string `gorm:"column:src_addr;uniqueIndex:unique_whitelist_net_flows" json:"src_addr"`
	LocalPort  string `gorm:"column:src_port;uniqueIndex:unique_whitelist_net_flows" json:"src_port"`
	RemoteAddr string `gorm:"column:dst_addr;uniqueIndex:unique_whitelist_net_flows" json:"dst_addr"`
	RemotePort string `gorm:"column:dst_port;uniqueIndex:unique_whitelist_net_flows" json:"dst_port"`
}

func (WhitelistNetFlow) TableName() string {
	return "whitelist_netflows_table"
}

// 初始化数据库连接并创建表
func InitBlacklistDatabase() {
	dsn := "host=/var/run/postgresql/ user=postgres password=postgres dbname=tc_cadet_dataset_db port=5432 sslmode=disable TimeZone=Asia/Shanghai"
	var err error
	DB, err = gorm.Open(postgres.Open(dsn), &gorm.Config{
		Logger: logger.Default.LogMode(logger.Silent),
	})
	if err != nil {
		log.Printf("Failed to connect to database: %v", err)
		log.Fatal("failed to connect database")
	}

	// delete all tables
	DB.Migrator().DropTable(
		&BlacklistSubject{},
		&BlacklistAction{},
		&BlacklistFile{},
		&BlacklistNetFlow{},
		&WhitelistSubject{},
		&WhitelistFile{},
		&WhitelistNetFlow{},
	)

	// 自动迁移创建所有表
	if err := DB.AutoMigrate(
		&BlacklistSubject{},
		&BlacklistAction{},
		&BlacklistFile{},
		&BlacklistNetFlow{},
		&WhitelistSubject{},
		&WhitelistFile{},
		&WhitelistNetFlow{},
	); err != nil {
		log.Printf("Failed to migrate tables: %v", err)
		log.Fatal("failed to migrate database")
	}

	log.Println("Database initialized successfully")
}
