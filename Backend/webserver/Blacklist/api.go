package Blacklist

import (
	"log"
	"net/http"

	"github.com/gin-gonic/gin"
)

// SetBlackList 设置黑名单
// 以json的格式传入黑名单数据，分为netflow、subject、object四种类型，分别是一个数组
// 例如：
/*
{
	"netflow": [
		{
			"src_addr": "192.168.1.1",
			"src_port": "8080",
			"dst_addr": "93.184.216.34",
			"dst_port": "443"
		}
	],
	"subject": [
		{
			"exec": ""
		}
	],
	"file": [
		{
			"path": ""
		}
	]
}
*/

type BlackList struct {
	NetFlow []BlacklistNetFlow `json:"netflow"`
	Subject []BlacklistSubject `json:"subject"`
	File    []BlacklistFile    `json:"file"`
}

func SetBlackList(c *gin.Context) {
	var blackList BlackList
	err := c.BindJSON(&blackList)
	if err != nil {
		log.Println(err)
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}
	//clear the table
	DB.Exec("TRUNCATE TABLE blacklist_netflows_table")
	for _, netFlow := range blackList.NetFlow {
		DB.Create(&netFlow)
	}
	DB.Exec("TRUNCATE TABLE blacklist_subjects_table")
	for _, subject := range blackList.Subject {
		DB.Create(&subject)
	}
	DB.Exec("TRUNCATE TABLE blacklist_files_table")
	for _, file := range blackList.File {
		DB.Create(&file)
	}
	c.JSON(http.StatusOK, gin.H{"message": "success"})
}

// GetBlackList 获取黑名单
// find all blacklist actions with flag = 0; send the blacklist actions to the frontend and set the flag to 1
func GetBlackList(c *gin.Context) {
	var blackListActions []BlacklistAction
	var blackListFlag1 []BlacklistAction
	DB.Where("flag = ?", 0).Find(&blackListActions)
	// apend 50 lateset actions with flag = 1
	DB.Where("flag = ?", 1).Order("timestamp_rec desc").Limit(50).Find(&blackListFlag1)
	blackListActions = append(blackListActions, blackListFlag1...)
	c.JSON(http.StatusOK, blackListActions)
	for _, blackListAction := range blackListActions {
		blackListAction.Flag = 1
		DB.Save(&blackListAction)
	}
}

func SetWhiteList(c *gin.Context) {
	var blackList BlackList
	err := c.BindJSON(&blackList)
	if err != nil {
		log.Println(err)
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}
	//clear the table

	c.JSON(http.StatusOK, gin.H{"message": "success"})
}
