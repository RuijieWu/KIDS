package open_api_forward

import (
	"net/http"
	"strconv"

	"github.com/gin-gonic/gin"
	"gitlab.ecloud.com/ecloud/ecloudsdkcscenter/model"
)

func ForwardLeakLinuxList(c *gin.Context) {
	// Ensure client is initialized only once
	clientOnce.Do(func() {
		// These values would typically come from a more secure source than the hardcoded values
		accessKey := myAccessKey
		secretKey := mySecretKey
		poolId := "CIDC-RP-25"

		client = createClient(accessKey, secretKey, poolId)
	})

	// 获取查询参数
	sizeStr := c.Query("size")
	pageStr := c.Query("page")

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

	// Prepare request with parameters
	request := &model.GetLeakLinuxListRequest{
		GetLeakLinuxListQuery: &model.GetLeakLinuxListQuery{
			Size: &size32,
			Page: &page32,
		},
	}

	// Call the API using the client
	response, err := client.GetLeakLinuxList(request)
	if err == nil {
		c.JSON(http.StatusOK, response)
	} else {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
	}
}

func ForwardLeakLinuxDetail(c *gin.Context) {
	// Ensure client is initialized only once
	clientOnce.Do(func() {
		// These values would typically come from a more secure source than the hardcoded values
		accessKey := myAccessKey
		secretKey := mySecretKey
		poolId := "CIDC-RP-25"

		client = createClient(accessKey, secretKey, poolId)
	})

	// 获取查询参数
	vulId := c.Query("vulId")

	request := &model.GetLeakLinuxDetailRequest{
		GetLeakLinuxDetailPath: &model.GetLeakLinuxDetailPath{
			VulId: &vulId,
		},
	}

	// Call the API using the client
	response, err := client.GetLeakLinuxDetail(request)

	if err == nil {
		c.JSON(http.StatusOK, response)
	} else {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
	}
}
