package main

import (
	"github.com/gin-gonic/gin"

	"KIDS/audit_data"
)

func main() {
	audit_data.InitDatabaseConnection()
	router := gin.Default()
	router.POST("/setup-audit", audit_data.SetupAudit)
	router.GET("/audit-logs", audit_data.GetAuditLogs)

	router.Run(":8080")
}
