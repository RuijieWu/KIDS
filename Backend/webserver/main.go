package main

import (
	"github.com/gin-gonic/gin"

	"KIDS/audit_data"
	"KIDS/open_api_forward"
)

func main() {
	audit_data.InitDatabaseConnection()
	router := gin.Default()
	router.POST("/data/setup-audit", audit_data.SetupAudit)
	router.GET("/data/audit-logs", audit_data.GetAuditLogs)

	//router.GET("/alarm/message/list", open_api_forward.ForwardMessageListRequest)

	router.Run(":8080")
}
