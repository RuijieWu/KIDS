<!-- SecurityAdvice.vue -->
<template>
    <div class="security-advice">
      <h1>安全建议页面</h1>
  
      <!-- 显示告警列表 -->
      <div v-if="warnings.length > 0">
        <h2>告警列表</h2>
        <ul>
          <li v-for="(warning, index) in warnings" :key="index">
            <p>告警时间: {{ warning.attackTimeStr }}</p>
            <p>告警类型: {{ warning.alarmType }}</p>
            <p>告警详情: {{ warning.resultDesc }}</p>
            <hr>
          </li>
        </ul>
      </div>
  
      <!-- 显示语言模型的回答 -->
      <div v-if="modelAnswer">
        <h2>语言模型回答</h2>
        <p>{{ modelAnswer }}</p>
      </div>
  
      <!-- 错误信息 -->
      <div v-if="error">
        <p>发生错误: {{ error }}</p>
      </div>
    </div>
  </template>
  
  <script>
  import axios from 'axios';
  
  export default {
    data() {
      return {
        warnings: [], // 存储从后端获取的告警列表数据
        modelAnswer: '', // 存储语言模型的回答
        error: '' // 存储错误信息
      };
    },
    mounted() {
      this.fetchWarningList();
    },
    methods: {
      async fetchWarningList() {
        try {
          // 发送请求获取告警列表数据
          const response = await axios.get('https://ecloud.10086.cn/api/safebox-asp/csproxy/v2/alarmCenter/alarmList', {
            params: {
              page: 1,
              pageSize: 10,
              // 其他参数根据需要添加
            }
            // headers: {
            //   'Authorization': 'Bearer your_token_here'
            // }
          });
  
          // 更新warnings数组
          this.warnings = response.data.body.content;
  
          // 将告警列表发送给语言模型API并获取回答
          this.sendToLanguageModel(this.warnings);
        } catch (error) {
          this.error = '获取告警列表失败: ' + error.message;
        }
      },
      async sendToLanguageModel(data) {
        try {
          // 构造需要发送给语言模型的数据格式，这里假设直接发送告警列表内容
          const requestData = {
            warnings: data
          };
  
          // 发送请求给语言模型的API接口
          const response = await axios.post('https://your-language-model-api-endpoint', requestData);
  
          // 更新页面上的语言模型回答
          this.modelAnswer = response.data.answer;
        } catch (error) {
          this.error = '请求语言模型失败: ' + error.message;
        }
      }
    }
  };
  </script>
  
  <style scoped>
  /* 可选：添加样式来美化页面 */
  .security-advice {
    max-width: 800px;
    margin: 0 auto;
    padding: 20px;
  }
  
  ul {
    list-style-type: none;
    padding: 0;
  }
  
  li {
    margin-bottom: 20px;
  }
  
  h1, h2 {
    color: #333;
  }
  </style>
  