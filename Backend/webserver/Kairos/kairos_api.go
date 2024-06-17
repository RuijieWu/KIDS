package Kairos

import (
	"errors"
	"log"
	"net/http"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"github.com/gin-gonic/gin"
)

func GetActions(c *gin.Context) {
	// 获取查询参数
	startTimeStr := c.Query("start_time")
	endTimeStr := c.Query("end_time")

	// 将时间字符串转换为 Unix 时间戳
	startTime, err := time.Parse("2006-01-02 15:04:05", startTimeStr)
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "Invalid start_time format"})
		return
	}
	endTime, err := time.Parse("2006-01-02 15:04:05", endTimeStr)
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "Invalid end_time format"})
		return
	}

	startTimeUnix := startTime.Unix()
	endTimeUnix := endTime.Unix()

	log.Printf("Received actions request with start time: %v, end time: %v\n", startTime, endTime)

	// 查询数据库
	var anomalousActions []AnomalousAction
	var dangerousActions []DangerousAction

	// 查询所有在时间段内的 AnomalousAction
	if err := DB.Where("time >= ? AND time <= ?", startTimeUnix, endTimeUnix).Find(&anomalousActions).Error; err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to query anomalous actions"})
		return
	}

	// 查询所有在时间段内的 DangerousAction
	if err := DB.Where("time >= ? AND time <= ?", startTimeUnix, endTimeUnix).Find(&dangerousActions).Error; err != nil {
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
	limitStr := c.Query("limit")

	// 将时间字符串转换为 Unix 时间戳
	startTime, err := time.Parse("2006-01-02 15:04:05", startTimeStr)
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "Invalid start_time format"})
		return
	}
	endTime, err := time.Parse("2006-01-02 15:04:05", endTimeStr)
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "Invalid end_time format"})
		return
	}

	startTimeUnix := startTime.Unix()
	endTimeUnix := endTime.Unix()

	// 解析 limit 参数
	limit, err := strconv.Atoi(limitStr)
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "Invalid limit parameter"})
		return
	}

	log.Printf("Received subjects request with start time: %v, end time: %v, limit: %d\n", startTime, endTime, limit)

	// 查询数据库
	var anomalousSubjects []AnomalousSubject
	var dangerousSubjects []DangerousSubject

	// 查询所有在时间段内的 AnomalousSubject
	if err := DB.Raw(`
		SELECT time, subject_type, subject_name, COUNT(*) as count
		FROM anomalous_subjects_table
		WHERE time >= ? AND time <= ?
		GROUP BY subject_type, subject_name
		ORDER BY count DESC
		LIMIT ?
	`, startTimeUnix, endTimeUnix, limit).Scan(&anomalousSubjects).Error; err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to query anomalous subjects"})
		return
	}

	// 查询所有在时间段内的 DangerousSubject
	if err := DB.Raw(`
		SELECT time, subject_type, subject_name, COUNT(*) as count
		FROM dangerous_subjects_table
		WHERE time >= ? AND time <= ?
		GROUP BY subject_type, subject_name
		ORDER BY count DESC
		LIMIT ?
	`, startTimeUnix, endTimeUnix, limit).Scan(&dangerousSubjects).Error; err != nil {
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

func GetObjects(c *gin.Context) {
	// 获取查询参数
	startTimeStr := c.Query("start_time")
	endTimeStr := c.Query("end_time")
	limitStr := c.Query("limit")

	// 将时间字符串转换为 Unix 时间戳
	startTime, err := time.Parse("2006-01-02 15:04:05", startTimeStr)
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "Invalid start_time format"})
		return
	}
	endTime, err := time.Parse("2006-01-02 15:04:05", endTimeStr)
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "Invalid end_time format"})
		return
	}

	startTimeUnix := startTime.Unix()
	endTimeUnix := endTime.Unix()

	// 解析 limit 参数
	limit, err := strconv.Atoi(limitStr)
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "Invalid limit parameter"})
		return
	}

	log.Printf("Received objects request with start time: %v, end time: %v, limit: %d\n", startTime, endTime, limit)

	// 查询数据库
	var anomalousObjects []AnomalousObject
	var dangerousObjects []DangerousObject

	// 查询所有在时间段内的 AnomalousObject
	if err := DB.Raw(`
		SELECT time, object_type, object_name, COUNT(*) as count
		FROM anomalous_objects_table
		WHERE time >= ? AND time <= ?
		GROUP BY object_type, object_name
		ORDER BY count DESC
		LIMIT ?
	`, startTimeUnix, endTimeUnix, limit).Scan(&anomalousObjects).Error; err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to query anomalous objects"})
		return
	}

	// 查询所有在时间段内的 DangerousObject
	if err := DB.Raw(`
		SELECT time, object_type, object_name, COUNT(*) as count
		FROM dangerous_objects_table
		WHERE time >= ? AND time <= ?
		GROUP BY object_type, object_name
		ORDER BY count DESC
		LIMIT ?
	`, startTimeUnix, endTimeUnix, limit).Scan(&dangerousObjects).Error; err != nil {
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

	// 将时间字符串转换为时间格式
	startTime, err := time.Parse("2006-01-02 15:04:05", startTimeStr)
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "Invalid start_time format"})
		return
	}
	endTime, err := time.Parse("2006-01-02 15:04:05", endTimeStr)
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "Invalid end_time format"})
		return
	}

	// 查询数据库，获取所有异常统计数据
	var results []AberrationStaticsTable
	if err := DB.Find(&results).Error; err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to query AberrationStaticsTable"})
		return
	}

	// 筛选在时间范围内的数据
	var filteredResults []AberrationStaticsTable
	for _, result := range results {
		// 解析 time_interval 字段
		timeIntervals := parseTimeInterval(result.TimeInterval)
		if timeIntervals == nil {
			continue
		}

		// 检查时间范围
		if timeIntervals[0].After(startTime) && timeIntervals[1].Before(endTime) {
			filteredResults = append(filteredResults, result)
		}
	}

	// 返回结果
	c.JSON(http.StatusOK, gin.H{
		"total": len(filteredResults),
		"data":  filteredResults,
	})
}

// 解析时间区间字符串为时间数组
func parseTimeInterval(intervalStr string) []time.Time {
	parts := strings.Split(intervalStr, "~")
	if len(parts) != 2 {
		return nil
	}

	startTime, err := time.Parse("2006-01-02 15:04:05.999999999", parts[0])
	if err != nil {
		return nil
	}

	endTime, err := time.Parse("2006-01-02 15:04:05.999999999", parts[1])
	if err != nil {
		return nil
	}

	return []time.Time{startTime, endTime}
}

func GetGraphVisual(c *gin.Context) {
	// 获取查询参数
	startTimeStr := c.Query("start_time")
	endTimeStr := c.Query("end_time")

	// 将时间字符串转换为时间格式
	startTime, err := time.Parse("2006-01-02 15:04:05", startTimeStr)
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "Invalid start_time format"})
		return
	}
	endTime, err := time.Parse("2006-01-02 15:04:05", endTimeStr)
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "Invalid end_time format"})
		return
	}

	// 指定的文件夹路径
	dir := "../../../artifact/graph_visual"

	// 获取目标文件夹中的所有文件
	files, err := filepath.Glob(dir + "/*.png")
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to read directory"})
		return
	}

	var results []string

	// 遍历文件，筛选符合时间范围的文件名
	for _, file := range files {
		// 提取文件名中的时间戳
		base := filepath.Base(file)
		prefix := strings.Split(base, ".")[0] // 去除后缀 .png

		// 检查时间范围
		fileStartTime, fileEndTime, err := parseTimestamp(prefix)
		if err != nil {
			continue
		}

		if fileStartTime.After(startTime) && fileEndTime.Before(endTime) {
			results = append(results, file)
		}
	}

	// 返回结果
	c.JSON(http.StatusOK, gin.H{
		"total": len(results),
		"data":  results,
	})
}

// 解析文件名中的时间戳范围
func parseTimestamp(prefix string) (startTime time.Time, endTime time.Time, err error) {
	parts := strings.Split(prefix, "~")
	if len(parts) != 2 {
		err = errors.New("invalid timestamp format")
		return
	}

	startTime, err = time.Parse("2006-01-02 15:04:05.999999999", parts[0])
	if err != nil {
		return
	}

	endTime, err = time.Parse("2006-01-02 15:04:05.999999999", parts[1])
	if err != nil {
		return
	}

	return startTime, endTime, nil
}
