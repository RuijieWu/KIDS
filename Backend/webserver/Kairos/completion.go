package Kairos

import (
	"bytes"
	"encoding/json"
	"net/http"
	"io"

	"github.com/gin-gonic/gin"
)

type Request struct {
	Model    string    `json:"model"`
	Messages []Message `json:"messages"`
}

type Message struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type Response struct {
	Completions []Completion `json:"completions"`
}

type Completion struct {
	Message struct {
		Role    string `json:"role"`
		Content string `json:"content"`
	} `json:"message"`
	Choices []struct {
		Text string `json:"text"`
	} `json:"choices"`
}

type UserRequest struct {
	Prompt string `json:"prompt"`
}

func GetCompletions(c *gin.Context) {
	var usrRequestData UserRequest
	if err := c.BindJSON(&usrRequestData); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "Invalid request payload"})
		return
	}

	apiURL := "https://open.bigmodel.cn/api/paas/v4/chat/completions"
	apiKey := myApiKey

	// 构建请求数据
	requestData := Request{
		Model: "glm-4",
		Messages: []Message{
			{
				Role:    "user",
				Content: "你是一名网络安全专家，正在调查一起网络攻击事件。请根据以下信息提供进一步的调查建议："+usrRequestData.Prompt,
			},
		},
	}

	// 将请求数据转换为JSON
	requestBody, err := json.Marshal(requestData)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to marshal request data"})
		return
	}

	// 创建HTTP请求
	req, err := http.NewRequest("POST", apiURL, bytes.NewBuffer(requestBody))
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to create HTTP request"})
		return
	}

	// 设置请求头
	req.Header.Set("Authorization", "Bearer "+apiKey)
	req.Header.Set("Content-Type", "application/json")

	// 发送HTTP请求
	client := &http.Client{}
	resp, err := client.Do(req)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to send HTTP request"})
		return
	}
	defer resp.Body.Close()

	// 读取响应数据
	responseBody, err := io.ReadAll(resp.Body)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to read response body"})
		return
	}

	// 解析响应数据
	var response Response
	err = json.Unmarshal(responseBody, &response)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to unmarshal response data"})
		return
	}

	// 提取结果
	var completions []string
	for _, completion := range response.Completions {
		completions = append(completions, completion.Message.Content)
	}

	c.JSON(http.StatusOK, gin.H{"completions": completions})
}