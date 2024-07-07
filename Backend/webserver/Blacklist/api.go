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
			"local_addr": "",
			"local_port": "",
			"remote_addr": "",
			"remote_port": ""
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
	for _, netFlow := range blackList.NetFlow {
		DB.Create(&netFlow)
	}
	for _, subject := range blackList.Subject {
		DB.Create(&subject)
	}
	for _, file := range blackList.File {
		DB.Create(&file)
	}
	c.JSON(http.StatusOK, gin.H{"message": "success"})
}

// GetBlackList 获取黑名单
// find all blacklist actions with flag = 0; send the blacklist actions to the frontend and set the flag to 1
func GetBlackList(c *gin.Context) {
	var blackListActions []BlacklistAction
	DB.Where("flag = ?", 0).Find(&blackListActions)
	c.JSON(http.StatusOK, blackListActions)
	for _, blackListAction := range blackListActions {
		blackListAction.Flag = 1
		DB.Save(&blackListAction)
	}
}