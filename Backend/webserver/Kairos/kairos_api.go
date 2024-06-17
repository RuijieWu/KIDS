package Kairos

import (
	"log"
	"net/http"
	"strconv"
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
