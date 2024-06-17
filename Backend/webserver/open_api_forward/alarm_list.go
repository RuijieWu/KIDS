package open_api_forward

import (
	"log"
	"net/http"
	"strconv"
	"strings"

	"github.com/gin-gonic/gin"
	"gitlab.ecloud.com/ecloud/ecloudsdkcscenter/model"
)

func ForwardAlarmListRequest(c *gin.Context) {
	// Ensure client is initialized only once
	clientOnce.Do(func() {
		// These values would typically come from a more secure source than the hardcoded values
		accessKey := myAccessKey
		secretKey := mySecretKey
		poolId := "CIDC-RP-25"

		client = createClient(accessKey, secretKey, poolId)
	})

	// 获取查询参数
	sizeStr := c.Query("pageSize")
	pageStr := c.Query("page")
	securitysStr := c.Query("securitys")
	alarmTypesStr := c.Query("alarmTypes")
	attackEndTime := c.Query("attackEndTime")
	attackStartTime := c.Query("attackStartTime")

	size, err := strconv.ParseInt(sizeStr, 10, 32)
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "Invalid size"})
		return
	}
	size32 := int32(size)

	page, err := strconv.ParseInt(pageStr, 10, 32)
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "Invalid page"})
		return
	}
	page32 := int32(page)

	// alarmTypes is a <integer(int32)> array(multi)
	alarmTypes := []int32{}
	for _, v := range strings.Split(alarmTypesStr, ",") {
		alarmType, err := strconv.ParseInt(v, 10, 32)
		if err != nil {
			log.Printf("Invalid alarmTypes: %e", err)
			c.JSON(http.StatusBadRequest, gin.H{"error": "Invalid alarmTypes"})
			return
		}
		alarmTypes = append(alarmTypes, int32(alarmType))
	}

	securitys := []int32{}
	for _, v := range strings.Split(securitysStr, ",") {
		security, err := strconv.ParseInt(v, 10, 32)
		if err != nil {
			log.Printf("Invalid securitys: %e", err)
			c.JSON(http.StatusBadRequest, gin.H{"error": "Invalid securitys"})
			return
		}
		securitys = append(securitys, int32(security))
	}

	// Prepare request with parameters
	request := &model.AlarmListRequest{
		AlarmListQuery: &model.AlarmListQuery{
			PageSize:        &size32,
			Page:            &page32,
			AlarmTypes:      alarmTypes,
			AttackEndTime:   &attackEndTime,
			AttackStartTime: &attackStartTime,
			Securitys:       securitys,
		},
	}

	// Call the API using the client
	response, err := client.AlarmList(request)
	if err == nil {
		c.JSON(http.StatusOK, response)
	} else {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
	}
}
