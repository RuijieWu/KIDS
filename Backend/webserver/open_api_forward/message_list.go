package open_api_forward

import (
	"net/http"
	"sync"

	"github.com/gin-gonic/gin"
	"ecloud.gitlab.com/ecloud/ecloudsdkcore/config"
	"ecloud.gitlab.com/ecloud/ecloudsdkecs"
	"ecloud.gitlab.com/ecloud/ecloudsdkecs/model"
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
	var requestParams struct {
		Size int32 `json:"size"`
		Page int32 `json:"page"`
	}

	if err := c.ShouldBindJSON(&requestParams); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	// Call the separate function to handle the request
	handleMessageList(c, requestParams.Size, requestParams.Page)
}
