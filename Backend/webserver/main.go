package main

import (
	"github.com/gin-contrib/cors"
	"github.com/gin-gonic/gin"

	"KIDS/Blacklist"
	"KIDS/Kairos"
	"KIDS/audit_data"
	"KIDS/open_api_forward"
)

func main() {
	audit_data.InitDatabaseConnection()
	Kairos.InitKairosDatabase()
	Blacklist.InitBlacklistDatabase()

	go Blacklist.Cronjob()

	router := gin.Default()

	// 添加CORS中间件
	router.Use(cors.New(cors.Config{
		AllowOrigins:     []string{"*"}, // 允许所有来源
		AllowMethods:     []string{"GET", "POST", "PUT", "DELETE", "OPTIONS"},
		AllowHeaders:     []string{"Origin", "Content-Length", "Content-Type"},
		ExposeHeaders:    []string{"Content-Length"},
		AllowCredentials: true,
	}))

	router.POST("/data/setup-audit", audit_data.SetupAudit)
	router.GET("/data/audit-logs", audit_data.GetAuditLogs)
	router.GET("/data/agent-info", audit_data.GetAgentInfo)
	router.POST("/data/upload-log", audit_data.UploadLog)

	router.GET("/alarm/message/list", open_api_forward.ForwardMessageListRequest)
	router.GET("/alarm/alarm/list", open_api_forward.ForwardAlarmListRequest)

	router.GET("/leak/linux/list", open_api_forward.ForwardLeakLinuxList)
	router.GET("/leak/linux/detail", open_api_forward.ForwardLeakLinuxDetail)

	router.GET("/kairos/actions", Kairos.GetActions)
	router.GET("/kairos/subjects", Kairos.GetSubjects)
	router.GET("/kairos/objects", Kairos.GetObjects)
	router.GET("/kairos/aberration-statics", Kairos.GetAberrationStatics)
	router.GET("/kairos/graph-visual", Kairos.GetGraphVisual)

	router.POST("/blacklist/set-blacklist", Blacklist.SetBlackList)
	router.GET("/blacklist/get-blacklist", Blacklist.GetBlackList)

	router.Run(":8080")

	// Block forever
	select {}
}
