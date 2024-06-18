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

func datetimeToNSTimestamp(dtStr string) int64 {
	// 将字符串解析为 time.Time 对象
	layout := "2006-01-02 15:04:05"
	dt, err := time.Parse(layout, dtStr)
	if err != nil {
		log.Println("Error parsing datetime:", err)
		return 0
	}

	// 计算纳秒级时间戳
	sec := dt.UnixNano() / 1e9  // 秒级时间戳
	nsec := dt.UnixNano() % 1e9 // 纳秒部分

	// 合并秒级时间戳和纳秒部分为整数类型的纳秒级时间戳
	nanoTimestamp := sec*1e9 + int64(nsec)
	log.Printf("Converted datetime %v to nanosecond timestamp %v\n", dt, nanoTimestamp)
	return nanoTimestamp
}

func GetActions(c *gin.Context) {
	// 获取查询参数
	startTimeStr := c.Query("start_time")
	endTimeStr := c.Query("end_time")

	startTimeUnix := datetimeToNSTimestamp(startTimeStr)
	endTimeUnix := datetimeToNSTimestamp(endTimeStr)

	log.Printf("Received actions request with start time: %v, end time: %v\n", startTimeUnix, endTimeUnix)

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
	limitStr := c.DefaultQuery("limit", "5") // 默认 limit 为 5

	// 将时间字符串转换为 Unix 时间戳
	startTimeUnix := datetimeToNSTimestamp(startTimeStr)
	endTimeUnix := datetimeToNSTimestamp(endTimeStr)

	// 解析 limit 参数
	limit, err := strconv.Atoi(limitStr)
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "Invalid limit parameter"})
		return
	}

	log.Printf("Received subjects request with start time: %v, end time: %v, limit: %d\n", startTimeUnix, endTimeUnix, limit)

	// 查询数据库
	var anomalousSubjects []AnomalousSubject
	var dangerousSubjects []DangerousSubject

	// 查询所有在时间段内的 AnomalousSubject
	if err := DB.Model(&AnomalousSubject{}).
		Select("time, subject_type, subject_name, COUNT(*) as count").
		Where("time >= ? AND time <= ?", startTimeUnix, endTimeUnix).
		Group("time, subject_type, subject_name").
		Order("count DESC").
		Limit(limit).
		Find(&anomalousSubjects).Error; err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to query anomalous subjects"})
		return
	}

	// 查询所有在时间段内的 DangerousSubject
	if err := DB.Model(&DangerousSubject{}).
		Select("time, subject_type, subject_name, COUNT(*) as count").
		Where("time >= ? AND time <= ?", startTimeUnix, endTimeUnix).
		Group("time, subject_type, subject_name").
		Order("count DESC").
		Limit(limit).
		Find(&dangerousSubjects).Error; err != nil {
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
	limitStr := c.DefaultQuery("limit", "5") // 默认 limit 为 5

	startTime := datetimeToNSTimestamp(startTimeStr)
	endTime := datetimeToNSTimestamp(endTimeStr)

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
	if err := DB.Model(&AnomalousObject{}).
		Select("time, object_type, object_name, COUNT(*) as count").
		Where("time >= ? AND time <= ?", startTime, endTime).
		Group("time, object_type, object_name").
		Order("count DESC").
		Limit(limit).
		Find(&anomalousObjects).Error; err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to query anomalous objects"})
		return
	}

	// 查询所有在时间段内的 DangerousObject
	if err := DB.Model(&DangerousObject{}).
		Select("time, object_type, object_name, COUNT(*) as count").
		Where("time >= ? AND time <= ?", startTime, endTime).
		Group("time, object_type, object_name").
		Order("count DESC").
		Limit(limit).
		Find(&dangerousObjects).Error; err != nil {
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

	startTimeUnix := datetimeToNSTimestamp(startTimeStr)
	endTimeUnix := datetimeToNSTimestamp(endTimeStr)

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

func GetGraphVisual(c *gin.Context) {
	// 获取查询参数
	startTimeStr := c.Query("start_time")
	endTimeStr := c.Query("end_time")

	startTime := datetimeToNSTimestamp(startTimeStr)
	endTime := datetimeToNSTimestamp(endTimeStr)

	// 指定的文件夹路径
	dir := "../../../artifact/graph_visual"

	// 获取目标文件夹中的所有文件
	files, err := filepath.Glob(dir + "/*.png")
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to read directory"})
		return
	}

	var results []gin.H

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

		if fileStartTime >= startTime && fileEndTime <= endTime {
			// 查询在这个时间范围内的 Anomalous 和 Dangerous 的 actions 数量
			var anomalousActionCount, dangerousActionCount int64
			if err := DB.Model(&AnomalousAction{}).
				Where("time >= ? AND time <= ?", fileStartTime, fileEndTime).
				Count(&anomalousActionCount).Error; err != nil {
				c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to count anomalous actions"})
				return
			}
			if err := DB.Model(&DangerousAction{}).
				Where("time >= ? AND time <= ?", fileStartTime, fileEndTime).
				Count(&dangerousActionCount).Error; err != nil {
				c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to count dangerous actions"})
				return
			}

			// 查询在这个时间范围内的 Anomalous 和 Dangerous 的 subjects 数量
			var anomalousSubjectCount, dangerousSubjectCount int64
			if err := DB.Model(&AnomalousSubject{}).
				Where("time >= ? AND time <= ?", fileStartTime, fileEndTime).
				Count(&anomalousSubjectCount).Error; err != nil {
				c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to count anomalous subjects"})
				return
			}
			if err := DB.Model(&DangerousSubject{}).
				Where("time >= ? AND time <= ?", fileStartTime, fileEndTime).
				Count(&dangerousSubjectCount).Error; err != nil {
				c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to count dangerous subjects"})
				return
			}

			// 查询在这个时间范围内的 Anomalous 和 Dangerous 的 objects 数量
			var anomalousObjectCount, dangerousObjectCount int64
			if err := DB.Model(&AnomalousObject{}).
				Where("time >= ? AND time <= ?", fileStartTime, fileEndTime).
				Count(&anomalousObjectCount).Error; err != nil {
				c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to count anomalous objects"})
				return
			}
			if err := DB.Model(&DangerousObject{}).
				Where("time >= ? AND time <= ?", fileStartTime, fileEndTime).
				Count(&dangerousObjectCount).Error; err != nil {
				c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to count dangerous objects"})
				return
			}

			// 构造结果
			result := gin.H{
				"file_name":                 base,
				"anomalous_action_count":    anomalousActionCount,
				"dangerous_action_count":    dangerousActionCount,
				"anomalous_subject_count":   anomalousSubjectCount,
				"dangerous_subject_count":   dangerousSubjectCount,
				"anomalous_object_count":    anomalousObjectCount,
				"dangerous_object_count":    dangerousObjectCount,
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
	parts := strings.Split(prefix, "~")
	if len(parts) != 2 {
		err = errors.New("invalid timestamp format")
		return
	}

	startTime, err = strconv.ParseInt(parts[0], 10, 64)
	if err != nil {
		return 0, 0, err
	}

	endTime, err = strconv.ParseInt(parts[1], 10, 64)
	if err != nil {
		return 0, 0, err
	}

	return startTime, endTime, nil
}
