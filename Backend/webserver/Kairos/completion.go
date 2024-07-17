package Kairos

import (
	"bytes"
	"encoding/json"
	"io"
	"log"
	"net/http"

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

type Choice struct {
	FinishReason string  `json:"finish_reason"`
	Index        int     `json:"index"`
	Message      Message `json:"message"`
}

type CompletionResponse struct {
	Choices []Choice `json:"choices"`
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
				Content: "你是一名网络安全专家，正在调查一起网络攻击事件。请根据以下信息提供进一步的调查建议（尽量简短一点）：" + usrRequestData.Prompt,
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

	log.Printf("Request data: %v", string(requestBody))

	// 发送HTTP请求
	client := &http.Client{}
	resp, err := client.Do(req)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to send HTTP request"})
		return
	}
	defer resp.Body.Close()

	var responseBody []byte
	if resp.StatusCode == http.StatusOK {
		// 读取响应体
		responseBody, err = io.ReadAll(resp.Body)
		if err != nil {
			c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to read response body"})
			return
		}
	} else {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to get response"})
		return
	}

	// Parse the completion response JSON
	var completionResponse CompletionResponse
	if err := json.Unmarshal(responseBody, &completionResponse); err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to parse completion response"})
		return
	}

	// Extract the advice content from the response
	advice := completionResponse.Choices[0].Message.Content

	// Return the advice to the client
	c.JSON(http.StatusOK, gin.H{"advice": advice})
}
