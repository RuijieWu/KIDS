package Blacklist

import (
    "context"
    "fmt"
    "log"
    "net/http"

    "github.com/gin-gonic/gin"
    "github.com/beltran/gohive"
)

type BlackList struct {
    NetFlow []BlacklistNetFlow `json:"netflow"`
    Subject []BlacklistSubject `json:"subject"`
    File    []BlacklistFile    `json:"file"`
}

var hiveConn *gohive.Connection

func SetBlackList(c *gin.Context) {
    var blackList BlackList
    err := c.BindJSON(&blackList)
    if err != nil {
        log.Println(err)
        c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
        return
    }

    ctx := context.Background()
    cursor := hiveConn.Cursor()

    // 清空 Hive 表
    cursor.Exec(ctx, "TRUNCATE TABLE blacklist_netflows_table")
    if cursor.Err != nil {
        log.Printf("Failed to truncate blacklist_netflows_table: %v", cursor.Err)
        c.JSON(http.StatusInternalServerError, gin.H{"error": cursor.Err.Error()})
        return
    }

    for _, netFlow := range blackList.NetFlow {
        query := fmt.Sprintf(
            `INSERT INTO blacklist_netflows_table (src_addr, src_port, dst_addr, dst_port) VALUES ('%s', '%s', '%s', '%s')`,
            netFlow.LocalAddr, netFlow.LocalPort, netFlow.RemoteAddr, netFlow.RemotePort,
        )
        cursor.Exec(ctx, query)
        if cursor.Err != nil {
            log.Printf("Failed to insert into blacklist_netflows_table: %v", cursor.Err)
            c.JSON(http.StatusInternalServerError, gin.H{"error": cursor.Err.Error()})
            return
        }
    }

    cursor.Exec(ctx, "TRUNCATE TABLE blacklist_subjects_table")
    if cursor.Err != nil {
        log.Printf("Failed to truncate blacklist_subjects_table: %v", cursor.Err)
        c.JSON(http.StatusInternalServerError, gin.H{"error": cursor.Err.Error()})
        return
    }

    for _, subject := range blackList.Subject {
        query := fmt.Sprintf(
            `INSERT INTO blacklist_subjects_table (exec) VALUES ('%s')`,
            subject.Exec,
        )
        cursor.Exec(ctx, query)
        if cursor.Err != nil {
            log.Printf("Failed to insert into blacklist_subjects_table: %v", cursor.Err)
            c.JSON(http.StatusInternalServerError, gin.H{"error": cursor.Err.Error()})
            return
        }
    }

    cursor.Exec(ctx, "TRUNCATE TABLE blacklist_files_table")
    if cursor.Err != nil {
        log.Printf("Failed to truncate blacklist_files_table: %v", cursor.Err)
        c.JSON(http.StatusInternalServerError, gin.H{"error": cursor.Err.Error()})
        return
    }

    for _, file := range blackList.File {
        query := fmt.Sprintf(
            `INSERT INTO blacklist_files_table (path) VALUES ('%s')`,
            file.Path,
        )
        cursor.Exec(ctx, query)
        if cursor.Err != nil {
            log.Printf("Failed to insert into blacklist_files_table: %v", cursor.Err)
            c.JSON(http.StatusInternalServerError, gin.H{"error": cursor.Err.Error()})
            return
        }
    }

    cursor.Close()
    c.JSON(http.StatusOK, gin.H{"message": "success"})
}

// GetBlackList 获取黑名单
// find all blacklist actions with flag = 0; send the blacklist actions to the frontend and set the flag to 1
func GetBlackList(c *gin.Context) {
	var blackListActions []BlacklistAction
	DB.Where("flag = ?", 0).Find(&blackListActions)
	c.JSON(http.StatusOK, blackListActions)
	for _, blackListAction := range blackListActions {
		blackListAction.Flag = 1
		DB.Save(&blackListAction)
	}
}