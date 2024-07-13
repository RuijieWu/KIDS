package open_api_forward

import (
	"log"
	"net/http"
	"strconv"
	"sync"

	"github.com/gin-gonic/gin"
	"gitlab.ecloud.com/ecloud/ecloudsdkcore/config"
	"gitlab.ecloud.com/ecloud/ecloudsdkcscenter"
	"gitlab.ecloud.com/ecloud/ecloudsdkcscenter/model"
)

var (
	client     *ecloudsdkcscenter.Client
	clientOnce sync.Once
)

// Encapsulate client creation with provided accessKey, secretKey, and poolId
func createClient(accessKey, secretKey, poolId string) *ecloudsdkcscenter.Client {
	config := &config.Config{
		AccessKey: &accessKey,
		SecretKey: &secretKey,
		PoolId:    &poolId,
	}
	return ecloudsdkcscenter.NewClient(config)
}

// Function to handle message list request with provided parameters
func handleMessageList(c *gin.Context, size, page int32) {
	// Ensure client is initialized only once
	clientOnce.Do(func() {
		// These values would typically come from a more secure source than the hardcoded values
		accessKey := myAccessKey
		secretKey := mySecretKey
		poolId := "CIDC-RP-25"

		client = createClient(accessKey, secretKey, poolId)
	})

	// Prepare request with parameters
	request := &model.MessageListRequest{
		MessageListQuery: &model.MessageListQuery{
			Size: &size,
			Page: &page,
		},
	}

	// Call the API using the client
	response, err := client.MessageList(request)
	if err == nil {
		c.JSON(http.StatusOK, response)
	} else {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
	}
}

func ForwardMessageListRequest(c *gin.Context) {
    // 获取查询参数
    sizeStr := c.Query("size")
    pageStr := c.Query("page")

    // 将查询参数转换为 int32
    size, err := strconv.Atoi(sizeStr)
    if err != nil {
        c.JSON(http.StatusBadRequest, gin.H{"error": "Invalid size parameter"})
        return
    }
    page, err := strconv.Atoi(pageStr)
    if err != nil {
        c.JSON(http.StatusBadRequest, gin.H{"error": "Invalid page parameter"})
        return
    }

    log.Printf("Received message list request with size: %d, page: %d\n", size, page)

    // 调用处理函数
    handleMessageList(c, int32(size), int32(page))
}

