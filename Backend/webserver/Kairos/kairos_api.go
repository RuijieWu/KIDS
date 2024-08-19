package Kairos

import (
	"encoding/base64"
	"errors"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"github.com/gin-gonic/gin"
)

func DatetimeToNSTimestamp(dtStr string) int64 {
	// 将字符串解析为 time.Time 对象
	layout := "2006-01-02 15:04:05"
	// 注意是new york时区
	loc, _ := time.LoadLocation("America/New_York")
	dt, _ := time.ParseInLocation(layout, dtStr, loc)
	return dt.UnixNano()
}

func GetActions(c *gin.Context) {
	// 获取查询参数
	startTimeStr := c.Query("start_time")
	endTimeStr := c.Query("end_time")

	startTimeUnix := DatetimeToNSTimestamp(startTimeStr)
	endTimeUnix := DatetimeToNSTimestamp(endTimeStr)

	log.Printf("Received actions request with start time: %v, end time: %v\n", startTimeUnix, endTimeUnix)

	// 查询数据库
	var anomalousActions []AnomalousAction = make([]AnomalousAction, 0)
	var dangerousActions []DangerousAction = make([]DangerousAction, 0)

	// 查询所有在时间段内的 AnomalousAction，使用 DISTINCT 去重
	if err := DB.Raw(`
        SELECT DISTINCT ON (time, subject_name, action, object_name) *
        FROM anomalous_actions_table
        WHERE time >= ? AND time <= ?`, startTimeUnix, endTimeUnix).Scan(&anomalousActions).Error; err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to query anomalous actions"})
		return
	}

	// 查询所有在时间段内的 DangerousAction，使用 DISTINCT 去重
	if err := DB.Raw(`
        SELECT DISTINCT ON (time, subject_name, action, object_name) *
        FROM dangerous_actions_table
        WHERE time >= ? AND time <= ?`, startTimeUnix, endTimeUnix).Scan(&dangerousActions).Error; err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to query dangerous actions"})
		return
	}

	// 返回结果
	c.JSON(http.StatusOK, gin.H{
		"anomalous_actions": gin.H{
			"total": len(anomalousActions),
			"data":  anomalousActions,
		},
		"dangerous_actions": gin.H{
			"total": len(dangerousActions),
			"data":  dangerousActions,
		},
	})
}

func GetGraphActions(c *gin.Context) {
	// 获取查询参数
	graphIndex := c.Query("graph_index")

	// 查询数据库
	var anomalousActions []AnomalousAction = make([]AnomalousAction, 0)
	var dangerousActions []DangerousAction = make([]DangerousAction, 0)

	// 查询所有在时间段内的 AnomalousAction，使用 DISTINCT 去重
	if err := DB.Raw(`
        SELECT DISTINCT ON (time, subject_name, action, object_name) *
        FROM anomalous_actions_table
        WHERE graph_index = ?`, graphIndex).Scan(&anomalousActions).Error; err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to query anomalous actions"})
		return
	}

	// 查询所有在时间段内的 DangerousAction，使用 DISTINCT 去重
	if err := DB.Raw(`
        SELECT DISTINCT ON (time, subject_name, action, object_name) *
        FROM dangerous_actions_table
        WHERE graph_index = ?`, graphIndex).Scan(&dangerousActions).Error; err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to query dangerous actions"})
		return
	}

	// 返回结果
	c.JSON(http.StatusOK, gin.H{
		"anomalous_actions": gin.H{
			"total": len(anomalousActions),
			"data":  anomalousActions,
		},
		"dangerous_actions": gin.H{
			"total": len(dangerousActions),
			"data":  dangerousActions,
		},
	})
}

func GetSubjects(c *gin.Context) {
	// 获取查询参数
	startTimeStr := c.Query("start_time")
	endTimeStr := c.Query("end_time")
	limitStr := c.DefaultQuery("limit", "99999") // 默认 limit 为 99999

	// 将时间字符串转换为 Unix 时间戳
	startTimeUnix := DatetimeToNSTimestamp(startTimeStr)
	endTimeUnix := DatetimeToNSTimestamp(endTimeStr)

	// 解析 limit 参数
	limit, err := strconv.Atoi(limitStr)
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "Invalid limit parameter"})
		return
	}

	log.Printf("Received subjects request with start time: %v, end time: %v, limit: %d\n", startTimeUnix, endTimeUnix, limit)

	// 查询数据库
	var anomalousSubjects []AnomalousSubject = make([]AnomalousSubject, 0)
	var dangerousSubjects []DangerousSubject = make([]DangerousSubject, 0)

	// 查询所有在时间段内的 AnomalousSubject，使用 DISTINCT 去重
	if err := DB.Raw(`
        SELECT DISTINCT ON (time, subject_name) *
        FROM anomalous_subjects_table
        WHERE time >= ? AND time <= ?
        ORDER BY time DESC
        LIMIT ?`, startTimeUnix, endTimeUnix, limit).Scan(&anomalousSubjects).Error; err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to query anomalous subjects"})
		return
	}

	// 查询所有在时间段内的 DangerousSubject，使用 DISTINCT 去重
	if err := DB.Raw(`
        SELECT DISTINCT ON (time, subject_name) *
        FROM dangerous_subjects_table
        WHERE time >= ? AND time <= ?
        ORDER BY time DESC
        LIMIT ?`, startTimeUnix, endTimeUnix, limit).Scan(&dangerousSubjects).Error; err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to query dangerous subjects"})
		return
	}

	// 返回结果
	c.JSON(http.StatusOK, gin.H{
		"anomalous_subjects": gin.H{
			"total": len(anomalousSubjects),
			"data":  anomalousSubjects,
		},
		"dangerous_subjects": gin.H{
			"total": len(dangerousSubjects),
			"data":  dangerousSubjects,
		},
	})
}

func GetGraphSubjects(c *gin.Context) {
	// 获取查询参数
	graphIndex := c.Query("graph_index")

	// 查询数据库
	var anomalousSubjects []AnomalousSubject = make([]AnomalousSubject, 0)
	var dangerousSubjects []DangerousSubject = make([]DangerousSubject, 0)

	// 查询所有在时间段内的 AnomalousSubject，使用 DISTINCT 去重
	if err := DB.Raw(`
        SELECT DISTINCT ON (time, subject_name) *
        FROM anomalous_subjects_table
        WHERE graph_index = ?`, graphIndex).Scan(&anomalousSubjects).Error; err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to query anomalous subjects"})
		return
	}

	// 查询所有在时间段内的 DangerousSubject，使用 DISTINCT 去重
	if err := DB.Raw(`
        SELECT DISTINCT ON (time, subject_name) *
        FROM dangerous_subjects_table
        WHERE graph_index = ?`, graphIndex).Scan(&dangerousSubjects).Error; err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to query dangerous subjects"})
		return
	}

	// 返回结果
	c.JSON(http.StatusOK, gin.H{
		"anomalous_subjects": gin.H{
			"total": len(anomalousSubjects),
			"data":  anomalousSubjects,
		},
		"dangerous_subjects": gin.H{
			"total": len(dangerousSubjects),
			"data":  dangerousSubjects,
		},
	})
}

func GetGraphObjects(c *gin.Context) {
	// 获取查询参数
	graphIndex := c.Query("graph_index")

	// 查询数据库
	var anomalousObjects []AnomalousObject = make([]AnomalousObject, 0)
	var dangerousObjects []DangerousObject = make([]DangerousObject, 0)

	// 查询所有在时间段内的 AnomalousObject，使用 DISTINCT 去重
	if err := DB.Raw(`
		SELECT DISTINCT ON (time, object_name) *
		FROM anomalous_objects_table
		WHERE graph_index = ?`, graphIndex).Scan(&anomalousObjects).Error; err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to query anomalous objects"})
		return
	}

	// 查询所有在时间段内的 DangerousObject，使用 DISTINCT 厍重
	if err := DB.Raw(`
		SELECT DISTINCT ON (time, object_name) *
		FROM dangerous_objects_table
		WHERE graph_index = ?`, graphIndex).Scan(&dangerousObjects).Error; err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to query dangerous objects"})
		return
	}

	// 返回结果
	c.JSON(http.StatusOK, gin.H{
		"anomalous_objects": gin.H{
			"total": len(anomalousObjects),
			"data":  anomalousObjects,
		},
		"dangerous_objects": gin.H{
			"total": len(dangerousObjects),
			"data":  dangerousObjects,
		},
	})
}

func GetObjects(c *gin.Context) {
	// 获取查询参数
	startTimeStr := c.Query("start_time")
	endTimeStr := c.Query("end_time")
	limitStr := c.DefaultQuery("limit", "99999") // 默认 limit 为 99999

	startTime := DatetimeToNSTimestamp(startTimeStr)
	endTime := DatetimeToNSTimestamp(endTimeStr)

	// 解析 limit 参数
	limit, err := strconv.Atoi(limitStr)
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "Invalid limit parameter"})
		return
	}

	log.Printf("Received objects request with start time: %v, end time: %v, limit: %d\n", startTime, endTime, limit)

	// 查询数据库
	var anomalousObjects []AnomalousObject = make([]AnomalousObject, 0)
	var dangerousObjects []DangerousObject = make([]DangerousObject, 0)

	// 查询所有在时间段内的 AnomalousObject，使用 DISTINCT 去重
	if err := DB.Raw(`
        SELECT DISTINCT ON (time, object_name) *
        FROM anomalous_objects_table
        WHERE time >= ? AND time <= ?
        ORDER BY time DESC
        LIMIT ?`, startTime, endTime, limit).Scan(&anomalousObjects).Error; err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to query anomalous objects"})
		return
	}

	// 查询所有在时间段内的 DangerousObject，使用 DISTINCT 去重
	if err := DB.Raw(`
        SELECT DISTINCT ON (time, object_name) *
        FROM dangerous_objects_table
        WHERE time >= ? AND time <= ?
        ORDER BY time DESC
        LIMIT ?`, startTime, endTime, limit).Scan(&dangerousObjects).Error; err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to query dangerous objects"})
		return
	}

	// 返回结果
	c.JSON(http.StatusOK, gin.H{
		"anomalous_objects": gin.H{
			"total": len(anomalousObjects),
			"data":  anomalousObjects,
		},
		"dangerous_objects": gin.H{
			"total": len(dangerousObjects),
			"data":  dangerousObjects,
		},
	})
}

func GetAberrationStatics(c *gin.Context) {
	// 获取查询参数
	startTimeStr := c.Query("start_time")
	endTimeStr := c.Query("end_time")

	startTimeUnix := DatetimeToNSTimestamp(startTimeStr)
	endTimeUnix := DatetimeToNSTimestamp(endTimeStr)

	var aberrationStatics []AberrationStaticsTable

	if err := DB.Model(&AberrationStaticsTable{}).
		Where("begin_time >= ? AND end_time <= ?", startTimeUnix, endTimeUnix).
		Find(&aberrationStatics).Error; err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to query aberration statics"})
		return
	}

	c.JSON(http.StatusOK, gin.H{
		"total": len(aberrationStatics),
		"data":  aberrationStatics,
	})
}

func GetGraphInfo(c *gin.Context) {
	// 获取查询参数
	startTimeStr := c.Query("start_time")
	endTimeStr := c.Query("end_time")

	startTime := DatetimeToNSTimestamp(startTimeStr)
	endTime := DatetimeToNSTimestamp(endTimeStr)

	// 指定的文件夹路径
	dir := "../../Engine/artifact/graph_visual"

	// 获取目标文件夹中的所有文件
	files, err := filepath.Glob(dir + "/*.png")
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to read directory"})
		return
	}

	var results []gin.H
	// TODO
	// 遍历文件，筛选符合时间范围的文件名
	for _, file := range files {
		// 提取文件名中的时间戳
		base := filepath.Base(file)
		prefix := strings.Split(base, ".png")[0] // 去除后缀 .png

		// 检查时间范围
		fileStartTime, fileEndTime, err := parseTimestamp(prefix)
		if err != nil {
			continue
		}

		graphIndex := strings.ReplaceAll(prefix, "", ":")
		log.Printf("Parsed prefix: %v\n", prefix)

		if fileStartTime >= startTime && fileEndTime <= endTime {
			// 查询数据库
			var anomalousActions []AnomalousAction = make([]AnomalousAction, 0)
			var dangerousActions []DangerousAction = make([]DangerousAction, 0)

			// 查询所有在时间段内的 AnomalousAction，使用 DISTINCT 去重
			if err := DB.Raw(`
				SELECT DISTINCT ON (time, subject_name, action, object_name) *
				FROM anomalous_actions_table
				WHERE graph_index = ?`, graphIndex).Scan(&anomalousActions).Error; err != nil {
				c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to query anomalous actions"})
				return
			}

			// 查询所有在时间段内的 DangerousAction，使用 DISTINCT 去重
			if err := DB.Raw(`
				SELECT DISTINCT ON (time, subject_name, action, object_name) *
				FROM dangerous_actions_table
				WHERE graph_index = ?`, graphIndex).Scan(&dangerousActions).Error; err != nil {
				c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to query dangerous actions"})
				return
			}

			// 查询所有在时间段内的 AnomalousSubject，使用 DISTINCT 去重
			var anomalousSubjects []AnomalousSubject = make([]AnomalousSubject, 0)
			var dangerousSubjects []DangerousSubject = make([]DangerousSubject, 0)

			// 查询所有在时间段内的 AnomalousSubject，使用 DISTINCT 去重
			if err := DB.Raw(`
				SELECT DISTINCT ON (time, subject_name) *
				FROM anomalous_subjects_table
				WHERE graph_index = ?`, graphIndex).Scan(&anomalousSubjects).Error; err != nil {
				c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to query anomalous subjects"})
				return
			}

			// 查询所有在时间段内的 DangerousSubject，使用 DISTINCT 去重
			if err := DB.Raw(`
				SELECT DISTINCT ON (time, subject_name) *
				FROM dangerous_subjects_table
				WHERE graph_index = ?`, graphIndex).Scan(&dangerousSubjects).Error; err != nil {
				c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to query dangerous subjects"})
				return
			}

			// 查询所有在时间段内的 AnomalousObject，使用 DISTINCT 去重
			var anomalousObjects []AnomalousObject = make([]AnomalousObject, 0)
			var dangerousObjects []DangerousObject = make([]DangerousObject, 0)

			// 查询所有在时间段内的 AnomalousObject，使用 DISTINCT 去重
			if err := DB.Raw(`
				SELECT DISTINCT ON (time, object_name) *
				FROM anomalous_objects_table
				WHERE graph_index = ?`, graphIndex).Scan(&anomalousObjects).Error; err != nil {
				c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to query anomalous objects"})
				return
			}

			// 查询所有在时间段内的 DangerousObject，使用 DISTINCT 厍重
			if err := DB.Raw(`
				SELECT DISTINCT ON (time, object_name) *
				FROM dangerous_objects_table
				WHERE graph_index = ?`, graphIndex).Scan(&dangerousObjects).Error; err != nil {
				c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to query dangerous objects"})
				return
			}

			// 构造结果
			result := gin.H{
				"file_name":               base,
				"anomalous_action_count":  len(anomalousActions),
				"dangerous_action_count":  len(dangerousActions),
				"anomalous_subject_count": len(anomalousSubjects),
				"dangerous_subject_count": len(dangerousSubjects),
				"anomalous_object_count":  len(anomalousObjects),
				"dangerous_object_count":  len(dangerousObjects),
			}
			results = append(results, result)
		}
	}

	// 返回结果
	c.JSON(http.StatusOK, gin.H{
		"total": len(results),
		"data":  results,
	})
}

// 解析文件名中的时间戳范围
func parseTimestamp(prefix string) (startTime int64, endTime int64, err error) {
	// 替换非标准字符
	correctedPrefix := strings.ReplaceAll(prefix, "", ":")

	// 去掉后缀 _6
	if strings.Contains(correctedPrefix, "_") {
		correctedPrefix = strings.Split(correctedPrefix, "_")[0]
	}

	parts := strings.Split(correctedPrefix, "~")
	if len(parts) != 2 {
		err = errors.New("invalid timestamp format")
		return
	}

	// 解析时间戳
	startTimeStr, endTimeStr := parts[0], parts[1]

	startTime = NsDatetimeToNSTimestamp(startTimeStr)
	endTime = NsDatetimeToNSTimestamp(endTimeStr)

	log.Printf("Parsed timestamp: %v ~ %v\n", startTime, endTime)

	return startTime, endTime, nil
}

func GetGraphContent(c *gin.Context) {
	// 获取查询参数
	fileName := c.Query("file_name")
	fileName = strings.ReplaceAll(fileName, ":", "")
	// 指定的文件夹路径
	dir := "../../Engine/artifact/graph_visual"
	file := filepath.Join(dir, fileName)
	log.Printf("Received graph content request for file: %v\n", file)

	// 读取文件内容并进行Base64编码
	fileContent, err := os.ReadFile(file)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to read file"})
		return
	}

	// 返回文件内容
	c.JSON(http.StatusOK, gin.H{
		"file_name":    fileName,
		"file_content": base64.StdEncoding.EncodeToString(fileContent),
	})
}

func NsDatetimeToNSTimestamp(dtStr string) int64 {
	// 将字符串解析为 time.Time 对象
	layout := "2006-01-02 15:04:05.999999999"
	loc, _ := time.LoadLocation("America/New_York")
	dt, err := time.ParseInLocation(layout, dtStr, loc)
	if err != nil {
		log.Println("Error parsing datetime:", err)
		return 0
	}

	// 合并秒级时间戳和纳秒部分为整数类型的纳秒级时间戳
	nanoTimestamp := dt.UnixNano()
	return nanoTimestamp
}
