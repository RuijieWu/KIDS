package Blacklist

import (
	"bytes"
	"encoding/json"
	"io"
	"io/ioutil"
	"log"
	"net/http"
	"time"

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
	NetFlow  []DangerousNetFlow  `json:"netflow"`
	Subject  []DangerousSubject  `json:"subject"`
	File     []DangerousFile     `json:"file"`
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
