package main

import (
	"github.com/gin-gonic/gin"

	"KIDS/Kairos"
	"KIDS/audit_data"
	"KIDS/open_api_forward"
)

func main() {
	audit_data.InitDatabaseConnection()
	Kairos.InitKairosDatabase()
	router := gin.Default()
	router.POST("/data/setup-audit", audit_data.SetupAudit)
	router.GET("/data/audit-logs", audit_data.GetAuditLogs)

	router.GET("/alarm/message/list", open_api_forward.ForwardMessageListRequest)
	router.GET("/alarm/alarm/list", open_api_forward.ForwardAlarmListRequest)

	router.GET("/leak/linux/list", open_api_forward.ForwardLeakLinuxList)
	router.GET("/leak/linux/detail", open_api_forward.ForwardLeakLinuxDetail)

	router.GET("/kairos/actions", Kairos.GetActions)
	router.GET("/kairos/subjects", Kairos.GetSubjects)
	router.GET("/kairos/objects", Kairos.GetObjects)
	router.Run(":8080")
}
