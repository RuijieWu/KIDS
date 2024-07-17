package audit_data

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"sync"
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

// AgentInfo represents the basic information of an agent
type AgentInfo struct {
	Status   string   `json:"status"`
	Uptime   string   `json:"uptime"`
	Paths    []string `json:"paths"`
	Hostname string   `json:"hostname"`
	HostIP   string   `json:"host_ip"`
}

// List of agent IPs
var agentIPs = []string{
	"localhost:8010",
	"localhost:8020",
	// "localhost:8012",
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

	var wg sync.WaitGroup
	results := make([]ProxyResponse, len(agentIPs))
	errors := make([]error, len(agentIPs))

	for i, ip := range agentIPs {
		wg.Add(1)
		go func(i int, ip string) {
			defer wg.Done()
			agentURL := "http://" + ip + "/setup-audit"
			resp, err := http.Post(agentURL, "application/json", bytes.NewBuffer(jsonData))
			if err != nil {
				errors[i] = err
				return
			}
			defer resp.Body.Close()

			body, err := io.ReadAll(resp.Body)
			if err != nil {
				errors[i] = err
				return
			}

			var proxyResponse ProxyResponse
			if err := json.Unmarshal(body, &proxyResponse); err != nil {
				errors[i] = err
				return
			}

			results[i] = proxyResponse
		}(i, ip)
	}

	wg.Wait()

	err_count := 0

	// Check for errors
	for _, err := range errors {
		if err != nil {
			err_count++
		}
	}

	if err_count == len(agentIPs) {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to communicate with all agents"})
		return
	}

	if err_count > 0 {
		c.JSON(http.StatusInternalServerError, gin.H{"error": fmt.Sprintf("Failed to communicate with %d agents", err_count)})
		return
	}

	c.JSON(http.StatusOK, results[0])
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

	var wg sync.WaitGroup
	results := make([]map[string]interface{}, len(agentIPs))
	errors := make([]error, len(agentIPs))

	for i, ip := range agentIPs {
		wg.Add(1)
		go func(i int, ip string) {
			defer wg.Done()
			agentURL := "http://" + ip + "/audit-logs"
			resp, err := http.Get(agentURL + "?start_time=" + startTime.Format("2006-01-02T15:04:05") + "&end_time=" + endTime.Format("2006-01-02T15:04:05"))
			if err != nil {
				errors[i] = err
				return
			}
			defer resp.Body.Close()

			body, err := io.ReadAll(resp.Body)
			if err != nil {
				errors[i] = err
				return
			}

			var result map[string]interface{}
			if err := json.Unmarshal(body, &result); err != nil {
				errors[i] = err
				return
			}

			results[i] = result

			// parse to type Events struct and insert to database
			events := Events{}
			if err := json.Unmarshal(body, &events); err != nil {
				log.Println("Error: ", err)
				errors[i] = err
				return
			}
			InsertEvents(events)
		}(i, ip)
	}

	wg.Wait()

	// Check for errors
	for _, err := range errors {
		if err != nil {
			c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to communicate with all agents"})
			return
		}
	}

	if len(results) == 0 {
		c.JSON(http.StatusOK, gin.H{"message": "Audit logs cleared, audit rules added, and auditd service restarted successfully."})
	}
	c.JSON(http.StatusOK, results)
}

func GetAgentInfo(c *gin.Context) {
	var wg sync.WaitGroup
	results := make([]AgentInfo, len(agentIPs))
	errors := make([]error, len(agentIPs))

	for i, ip := range agentIPs {
		wg.Add(1)
		go func(i int, ip string) {
			defer wg.Done()
			infoURL := "http://" + ip + "/info"
			resp, err := http.Get(infoURL)
			if err != nil {
				errors[i] = err
				return
			}
			defer resp.Body.Close()

			body, err := io.ReadAll(resp.Body)
			if err != nil {
				errors[i] = err
				return
			}

			var agentInfo AgentInfo
			if err := json.Unmarshal(body, &agentInfo); err != nil {
				errors[i] = err
				return
			}

			results[i] = agentInfo
		}(i, ip)
	}

	wg.Wait()

	// Check for errors
	for _, err := range errors {
		if err != nil {
			c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to communicate with all agents"})
			return
		}
	}

	c.JSON(http.StatusOK, results)
}

// read a 'json' type Events struct from any type of file and insert to database
func UploadLog(c *gin.Context) {
	file, err := c.FormFile("file")
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	src, err := file.Open()
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}
	defer src.Close()

	// parse to type Events struct and insert to database
	events := Events{}
	if err := json.NewDecoder(src).Decode(&events); err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}
	InsertEvents(events)

	c.JSON(http.StatusOK, gin.H{"message": "Successfully uploaded and inserted the logs"})
}
