package audit_data

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"os/exec"
	"strconv"
	"strings"
	"sync"
	"time"
	"unicode"

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
}

var AgentIPs = agentIPs

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
			log.Println("Agent response:", resp)
			if err != nil {
				results[i] = ProxyResponse{Message: "dead"}
				errors[i] = err
				return
			}
			defer resp.Body.Close()

			body, err := io.ReadAll(resp.Body)
			if err != nil {
				results[i] = ProxyResponse{Message: "dead"}
				errors[i] = err
				return
			}

			var proxyResponse ProxyResponse
			if err := json.Unmarshal(body, &proxyResponse); err != nil {
				results[i] = ProxyResponse{Message: "dead"}
				errors[i] = err
				return
			}

			results[i] = proxyResponse
		}(i, ip)
	}

	wg.Wait()

	// Check for errors
	for _, err := range errors {
		if err != nil {
			log.Println("Error:", err)
		}
	}

	c.JSON(http.StatusOK, results)
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
			log.Println("Agent response:", resp)

			if err != nil {
				results[i] = map[string]interface{}{"status": "dead"}
				errors[i] = err
				return
			}
			defer resp.Body.Close()

			body, err := io.ReadAll(resp.Body)
			if err != nil {
				results[i] = map[string]interface{}{"status": "dead"}
				errors[i] = err
				return
			}
			log.Println("Body:", string(body))

			var result map[string]interface{}
			if err := json.Unmarshal(body, &result); err != nil {
				results[i] = map[string]interface{}{"status": "dead"}
				errors[i] = err
				return
			}
			log.Println("Result:", result)

			results[i] = result

			// parse to type Events struct and insert to database
			events := Events{}
			if err := json.Unmarshal(body, &events); err != nil {
				log.Println("Error:", err)
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
			log.Println("Error:", err)
		}
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
				results[i] = AgentInfo{Status: "dead", HostIP: ip}
				errors[i] = err
				return
			}
			defer resp.Body.Close()

			body, err := io.ReadAll(resp.Body)
			if err != nil {
				results[i] = AgentInfo{Status: "dead", HostIP: ip}
				errors[i] = err
				return
			}

			var agentInfo AgentInfo
			if err := json.Unmarshal(body, &agentInfo); err != nil {
				results[i] = AgentInfo{Status: "dead", HostIP: ip}
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
			log.Println("Error:", err)
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

	// log the file name and the size
	log.Println("Uploaded file:", file.Filename)
	log.Println("File size:", file.Size)

	// parse to type Events struct and insert to database
	events := Events{}
	if err := json.NewDecoder(src).Decode(&events); err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}
	InsertEvents(events)

	c.JSON(http.StatusOK, gin.H{"message": "Successfully uploaded and inserted the logs"})
}

// hdfs dfsadmin -report
// Configured Capacity: 105418559488 (98.18 GB)
// Present Capacity: 86294810624 (80.37 GB)
// DFS Remaining: 86294704128 (80.37 GB)
// DFS Used: 106496 (104 KB)
// DFS Used%: 0.00%
// Replicated Blocks:
//         Under replicated blocks: 9
//         Blocks with corrupt replicas: 0
//         Missing blocks: 0
//         Missing blocks (with replication factor 1): 0
//         Low redundancy blocks with highest priority to recover: 5
//         Pending deletion blocks: 0
// Erasure Coded Block Groups:
//         Low redundancy block groups: 0
//         Block groups with corrupt internal blocks: 0
//         Missing block groups: 0
//         Low redundancy blocks with highest priority to recover: 0
//         Pending deletion blocks: 0

// -------------------------------------------------
// Live datanodes (2):

// Name: 10.1.20.12:9866 (10.1.20.12)
// Hostname: hadoop001
// Decommission Status : Normal
// Configured Capacity: 63278178304 (58.93 GB)
// DFS Used: 28672 (28 KB)
// Non DFS Used: 7417516032 (6.91 GB)
// DFS Remaining: 53181169664 (49.53 GB)
// DFS Used%: 0.00%
// DFS Remaining%: 84.04%
// Configured Cache Capacity: 0 (0 B)
// Cache Used: 0 (0 B)
// Cache Remaining: 0 (0 B)
// Cache Used%: 100.00%
// Cache Remaining%: 0.00%
// Xceivers: 0
// Last contact: Sat Aug 10 21:38:23 CST 2024
// Last Block Report: Sat Aug 10 15:44:08 CST 2024
// Num of Blocks: 0

// Name: 106.52.25.152:9866 (106.52.25.152)
// Hostname: hadoop002
// Decommission Status : Normal
// Configured Capacity: 42140381184 (39.25 GB)
// DFS Used: 77824 (76 KB)
// Non DFS Used: 7206277120 (6.71 GB)
// DFS Remaining: 33113534464 (30.84 GB)
// DFS Used%: 0.00%
// DFS Remaining%: 78.58%
// Configured Cache Capacity: 0 (0 B)
// Cache Used: 0 (0 B)
// Cache Remaining: 0 (0 B)
// Cache Used%: 100.00%
// Cache Remaining%: 0.00%
// Xceivers: 0
// Last contact: Sat Aug 10 21:38:22 CST 2024
// Last Block Report: Sat Aug 10 17:32:40 CST 2024
// Num of Blocks: 5

// Dead datanodes (2):

// Name: 106.53.44.175:9866 (hadoop003)
// Hostname: hadoop003
// Decommission Status : Normal
// Configured Capacity: 42140381184 (39.25 GB)
// DFS Used: 28672 (28 KB)
// Non DFS Used: 7085801472 (6.60 GB)
// DFS Remaining: 33234059264 (30.95 GB)
// DFS Used%: 0.00%
// DFS Remaining%: 78.87%
// Configured Cache Capacity: 0 (0 B)
// Cache Used: 0 (0 B)
// Cache Remaining: 0 (0 B)
// Cache Used%: 100.00%
// Cache Remaining%: 0.00%
// Xceivers: 0
// Last contact: Wed Jul 31 11:13:27 CST 2024
// Last Block Report: Wed Jul 31 06:05:17 CST 2024
// Num of Blocks: 0

// Name: 43.138.200.89:9866 (43.138.200.89)
// Hostname: hadoop004
// Decommission Status : Normal
// Configured Capacity: 0 (0 B)
// DFS Used: 0 (0 B)
// Non DFS Used: 0 (0 B)
// DFS Remaining: 0 (0 B)
// DFS Used%: 100.00%
// DFS Remaining%: 0.00%
// Configured Cache Capacity: 0 (0 B)
// Cache Used: 0 (0 B)
// Cache Remaining: 0 (0 B)
// Cache Used%: 100.00%
// Cache Remaining%: 0.00%
// Xceivers: 0
// Last contact: Thu Aug 08 18:46:28 CST 2024
// Last Block Report: Thu Aug 08 15:48:59 CST 2024
// Num of Blocks: 0

// postgres-#     pg_stat_database;
//        datname       |  size   | xact_commit | xact_rollback | blks_read | blks_hit
// ---------------------+---------+-------------+---------------+-----------+----------
//                      |         |           0 |             0 |        11 |   173602
//  postgres            | 9517 kB |        4693 |             0 |       138 |   225318
//  tc_cadet_dataset_db | 9685 kB |       17721 |             4 |       398 |  1279211
//  template1           | 7877 kB |        6082 |             0 |        68 |   253483
//  template0           | 7729 kB |           0 |             0 |         0 |        0
// (5 rows)

type Node struct {
	NodeName     string `json:"node_name"`
	TotalSpace   string `json:"total_space"`
	UsedSpace    string `json:"used_space"`
	Transactions string `json:"transactions"`
	Rollbacks    string `json:"rollbacks"`
	BlksRead     string `json:"blks_read"`
	BlksHit      string `json:"blks_hit"`
}

type Storage struct {
	StorageName string `json:"storage_name"`
	TotalSpace  string `json:"total_space"`
	UsedSpace   string `json:"used_space"`
	Nodes       []Node `json:"nodes"`
}

type StorageReport struct {
	Warehouses []Storage `json:"warehouses"`
}

func fetchHDFSReport() Storage {
	output, err := exec.Command("hdfs", "dfsadmin", "-report").Output()
	if err != nil {
		fmt.Println("Error executing HDFS command:", err)
		return Storage{}
	}

	lines := strings.Split(string(output), "\n")
	var nodes []Node
	var warehouse Storage
	first := true

	for _, line := range lines {
		line = strings.TrimSpace(line)
		if first {
			if strings.HasPrefix(line, "Configured Capacity:") {
				warehouse.TotalSpace = strings.TrimSpace(strings.Split(line, ":")[1])
			} else if strings.HasPrefix(line, "DFS Used:") && len(nodes) == 0 {
				warehouse.UsedSpace = strings.TrimSpace(strings.Split(line, ":")[1])
				first = false
			}
		} else if strings.HasPrefix(line, "Name:") {
			node := Node{NodeName: strings.TrimSpace(strings.Split(line, ":")[1])}
			nodes = append(nodes, node)
		} else if strings.HasPrefix(line, "Configured Capacity:") && len(nodes) > 0 {
			nodes[len(nodes)-1].TotalSpace = strings.TrimSpace(strings.Split(line, ":")[1])
		} else if strings.HasPrefix(line, "DFS Used:") && len(nodes) > 0 {
			nodes[len(nodes)-1].UsedSpace = strings.TrimSpace(strings.Split(line, ":")[1])
		}
	}
	warehouse.Nodes = nodes
	warehouse.StorageName = "HDFS Storage"
	return warehouse
}

// ParseSize takes a size string like "9517 kB" and converts it to bytes
func ParseSize(sizeStr string) (int64, error) {
	// Split the string into value and unit
	parts := strings.Fields(sizeStr)
	if len(parts) != 2 {
		log.Println(sizeStr)
		return 0, fmt.Errorf("invalid size format")
	}

	// Convert the value part to a number
	value, err := strconv.ParseFloat(parts[0], 64)
	if err != nil {
		return 0, err
	}

	// Determine the unit multiplier
	var multiplier int64
	switch parts[1] {
	case "B":
		multiplier = 1
	case "KB":
		multiplier = 1024
	case "kB":
		multiplier = 1024
	case "k":
		multiplier = 1024
	case "K":
		multiplier = 1024
	case "mB":
		multiplier = 1024 * 1024
	case "MB":
		multiplier = 1024 * 1024
	case "m":
		multiplier = 1024 * 1024
	case "M":
		multiplier = 1024 * 1024
	case "gB":
		multiplier = 1024 * 1024 * 1024
	case "GB":
		multiplier = 1024 * 1024 * 1024
	case "g":
		multiplier = 1024 * 1024 * 1024
	case "G":
		multiplier = 1024 * 1024 * 1024
	default:
		return 0, fmt.Errorf("unknown unit")
	}

	// Calculate the size in bytes
	return int64(value * float64(multiplier)), nil
}

// FormatSize takes a size in bytes and converts it to a human-readable format
func FormatSize(sizeInBytes int64) string {
	// Define unit thresholds
	const (
		KB = 1024
		MB = KB * 1024
		GB = MB * 1024
	)

	// Determine the appropriate unit and format the output
	switch {
	case sizeInBytes >= GB:
		return fmt.Sprintf("%.2f GB", float64(sizeInBytes)/float64(GB))
	case sizeInBytes >= MB:
		return fmt.Sprintf("%.2f MB", float64(sizeInBytes)/float64(MB))
	case sizeInBytes >= KB:
		return fmt.Sprintf("%.2f kB", float64(sizeInBytes)/float64(KB))
	default:
		return fmt.Sprintf("%d B", sizeInBytes)
	}
}

func fetchPostgresReport() Storage {
	cmd := `PGPASSWORD='postgres' psql -U postgres -c "SELECT datname, pg_size_pretty(pg_database_size(datname)) AS size, 
            xact_commit, xact_rollback, blks_read, blks_hit 
            FROM pg_stat_database;"`
	output, err := exec.Command("bash", "-c", cmd).Output()
	if err != nil {
		fmt.Println("Error executing PostgreSQL command:", err)
		return Storage{}
	}

	lines := strings.Split(string(output), "\n")
	var nodes []Node

	for _, line := range lines {
		fields := strings.Fields(line)

		if len(fields) >= 10 && fields[0] != "datname" {
			node := Node{
				NodeName:     fields[0],
				UsedSpace:    fields[2] + " " + fields[3],
				TotalSpace:   fields[2] + " " + fields[3],
				Transactions: fields[5],
				Rollbacks:    fields[7],
				BlksRead:     fields[9],
				BlksHit:      fields[11],
			}
			nodes = append(nodes, node)
		}
	}
	totalUsedSpace := 0
	for i := range nodes {
		space, err := ParseSize(nodes[i].UsedSpace)
		if err != nil {
			fmt.Println("Error parsing size:", err)
			return Storage{}
		}
		totalUsedSpace += int(space)
	}

	// use df -h to get total space
	cmd = `df -h . | awk 'NR==2 {print $4}'`
	// 	(base) root@hadoop004:/home/ubuntu/softbei/KIDS/Backend/webserver# df -h
	// Filesystem      Size  Used Avail Use% Mounted on
	// udev            1.7G     0  1.7G   0% /dev
	// tmpfs           341M   35M  307M  11% /run
	// /dev/vda2        40G   32G  5.9G  85% /
	// tmpfs           1.7G   64K  1.7G   1% /dev/shm
	// tmpfs           5.0M     0  5.0M   0% /run/lock
	// tmpfs           1.7G     0  1.7G   0% /sys/fs/cgroup
	// (base) root@hadoop004:/home/ubuntu/softbei/KIDS/Backend
	output, err = exec.Command("bash", "-c", cmd).Output()
	if err != nil {
		fmt.Println("Error executing df command:", err)
		return Storage{}
	}
	//delete \n from output
	output = output[:len(output)-1]
	// 5.6G->5.6 G
	// 把output拆除成两部分，第一部分是数字，第二部分是单位
	output_, _ := FormatSizeString(string(output))

	outputInt, _ := ParseSize(string(output_))
	outputInt += int64(totalUsedSpace)
	outputStr := FormatSize(int64(outputInt))

	warehouse := Storage{
		StorageName: "Postgres Storage",
		TotalSpace:  outputStr,
		UsedSpace:   FormatSize(int64(totalUsedSpace)),
		Nodes:       nodes,
	}
	return warehouse
}

// FormatSizeString splits the size string and returns it in the format "number unit"
func FormatSizeString(sizeStr string) (string, error) {
	// Find the index where the unit starts (first non-digit character)
	var i int
	for i = 0; i < len(sizeStr); i++ {
		if !unicode.IsDigit(rune(sizeStr[i])) && sizeStr[i] != '.' {
			break
		}
	}

	// If no unit is found, return an error
	if i == len(sizeStr) {
		return "", fmt.Errorf("invalid size format: no unit found")
	}

	// Split the string into number and unit parts
	numberPart := sizeStr[:i]
	unitPart := sizeStr[i:]

	// Return the formatted string with a space between the number and unit
	return fmt.Sprintf("%s %s", numberPart, unitPart), nil
}

// cd /KIDS/server/hive
// bin/hiveserver2

func GetStats(c *gin.Context) {
	hdfsReport := fetchHDFSReport()
	postgresReport := fetchPostgresReport()

	response := StorageReport{
		Warehouses: []Storage{
			hdfsReport,
			postgresReport,
		},
	}

	c.JSON(200, response)
}
