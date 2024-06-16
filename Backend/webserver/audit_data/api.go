package audit_data

import (
	"bytes"
	"encoding/json"
	"io/ioutil"
	"net/http"
	"time"

	"github.com/gin-gonic/gin"
)

// DirectoryPaths represents the directory paths in the request body
type DirectoryPaths struct {
	Paths []string `json:"paths"`
}

// AuditLogsRequest represents the query parameters for the audit logs request
type AuditLogsRequest struct {
	StartTime string `form:"start_time" binding:"required"`
	EndTime   string `form:"end_time" binding:"required"`
}

// ProxyResponse is a helper to capture the response from the Python server
type ProxyResponse struct {
	Message string `json:"message"`
}

func SetupAudit(c *gin.Context) {
	var directoryPaths DirectoryPaths
	if err := c.ShouldBindJSON(&directoryPaths); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	jsonData, err := json.Marshal(directoryPaths)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	resp, err := http.Post("http://localhost:8000/setup-audit", "application/json", bytes.NewBuffer(jsonData))
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to connect to the Python service"})
		return
	}
	defer resp.Body.Close()

	body, err := ioutil.ReadAll(resp.Body)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to read response from the Python service"})
		return
	}

	var proxyResponse ProxyResponse
	if err := json.Unmarshal(body, &proxyResponse); err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to parse response from the Python service"})
		return
	}

	c.JSON(resp.StatusCode, proxyResponse)
}

func GetAuditLogs(c *gin.Context) {
	var auditLogsRequest AuditLogsRequest
	if err := c.ShouldBindQuery(&auditLogsRequest); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	startTime, err := time.Parse("2006-01-02T15:04:05", auditLogsRequest.StartTime)
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "Invalid start_time format. Use 'YYYY-MM-DDTHH:MM:SS'"})
		return
	}

	endTime, err := time.Parse("2006-01-02T15:04:05", auditLogsRequest.EndTime)
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "Invalid end_time format. Use 'YYYY-MM-DDTHH:MM:SS'"})
		return
	}

	pythonURL := "http://localhost:8000/audit-logs"
	resp, err := http.Get(pythonURL + "?start_time=" + startTime.Format("2006-01-02T15:04:05") + "&end_time=" + endTime.Format("2006-01-02T15:04:05"))
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to connect to the Python service"})
		return
	}
	defer resp.Body.Close()

	body, err := ioutil.ReadAll(resp.Body)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to read response from the Python service"})
		return
	}

	var result map[string]interface{}
	if err := json.Unmarshal(body, &result); err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to parse response from the Python service"})
		return
	}

	// parse to type Events struct
	events := Events{}
	if err := json.Unmarshal(body, &events); err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to parse response from the Python service"})
		return
	}

	InsertEvents(events)

	c.JSON(resp.StatusCode, result)
}