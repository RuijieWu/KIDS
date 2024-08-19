package Kairos

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"

	"github.com/gin-gonic/gin"
)

const (
	engineHost = "localhost"
	enginePort = "8000"
)

type KairosResp struct {
	Status string                 `json:"Status"`
	Msg    string                 `json:"Msg"`
	Config map[string]interface{} `json:"config,omitempty"`
}

// /ping
func HandlePing(c *gin.Context) {
	resp, err := http.Get(fmt.Sprintf("http://%s:%s/ping", engineHost, enginePort))
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to reach KIDS Engine"})
		return
	}
	defer resp.Body.Close()

	body, _ := io.ReadAll(resp.Body)
	var KairosResp KairosResp
	json.Unmarshal(body, &KairosResp)

	c.JSON(resp.StatusCode, KairosResp)
}

// /api/<cmd>/<begin_time>/<end_time>
func HandleApi(c *gin.Context) {
	cmd := c.Param("cmd")
	beginTime := c.Param("begin_time")
	endTime := c.Param("end_time")

	url := fmt.Sprintf("http://%s:%s/api/%s/%s/%s", engineHost, enginePort, cmd, beginTime, endTime)
	resp, err := http.Get(url)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to reach KIDS Engine"})
		return
	}
	defer resp.Body.Close()

	body, _ := io.ReadAll(resp.Body)
	var KairosResp KairosResp
	json.Unmarshal(body, &KairosResp)

	c.JSON(resp.StatusCode, KairosResp)
}

// /config/update/<key>/<value>
func HandleConfigUpdate(c *gin.Context) {
	key := c.Param("key")
	value := c.Param("value")

	url := fmt.Sprintf("http://%s:%s/config/update/%s/%s", engineHost, enginePort, key, value)
	resp, err := http.Get(url)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to reach KIDS Engine"})
		return
	}
	defer resp.Body.Close()

	body, _ := io.ReadAll(resp.Body)
	var KairosResp KairosResp
	json.Unmarshal(body, &KairosResp)

	c.JSON(resp.StatusCode, KairosResp)
}

// /config/view
func HandleConfigView(c *gin.Context) {
	resp, err := http.Get(fmt.Sprintf("http://%s:%s/config/view", engineHost, enginePort))
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to reach KIDS Engine"})
		return
	}
	defer resp.Body.Close()

	body, _ := io.ReadAll(resp.Body)
	var KairosResp KairosResp
	json.Unmarshal(body, &KairosResp)

	c.JSON(resp.StatusCode, KairosResp)
}
