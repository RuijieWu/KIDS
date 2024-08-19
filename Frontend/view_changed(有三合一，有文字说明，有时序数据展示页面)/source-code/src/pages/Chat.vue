<template>
    <div class="chat-container">
      <div class="chat-messages" ref="chatMessages">
        <div v-for="(message, index) in messages" :key="index" 
             :class="['message', message.type]">
          <img v-if="message.type === 'ai'" class="message-avatar" src="../assets/img/k-logo.png" alt="AI Avatar">
          <div class="message-content" v-if="message.type === 'user'">
            {{ message.content }}
          </div>
          <div class="message-content" v-else>
            {{ message.typedText }}
          </div>
          <img v-if="message.type === 'user'" class="message-avatar" src="../assets/img/new_logo.png" alt="User Avatar">
        </div>
      </div>
      <div class="default-messages" v-if="showDefaultMessages">
        <button class="ml-4" v-for="(message, index) in defaultMessages" :key="index" @click="selectDefaultMessage(message)">
          {{ message }}
        </button>
      </div>
      <div class="chat-input">
        <el-input
          v-model="userInput"
          placeholder="输入消息..."
          @keyup.enter.native="sendMessage"
        >
          <el-button slot="append" @click="sendMessage" icon="el-icon-s-promotion">
            发送
          </el-button>
        </el-input>
      </div>
    </div>
  </template>
  
  <script>
  import axios from 'axios';
  import { mapGetters } from 'vuex';
  export default {
    data() {
      return {
        messages: [],
        userInput: '',
        showDefaultMessages: true, // 控制默认气泡的显示
        defaultMessages: [
        ]
      }
    },
    computed: {
    ...mapGetters(['getCurrentSecurityEvent'])
  },
    methods: {
      typeNextChar(message) {
        if (message.typeIndex < message.content.length) {
          message.typedText += message.content.charAt(message.typeIndex)
          message.typeIndex++
          setTimeout(() => this.typeNextChar(message), 50)
        }
      },
      sendMessage() {
        if (this.userInput.trim() === '') return
  
        // 隐藏默认气泡
        if (this.showDefaultMessages) {
          this.showDefaultMessages = false
        }
  
        const userMessage = { type: 'user', content: this.userInput }
        this.addMessage(userMessage)
        const userInput = this.userInput
        this.userInput = ''
        const prompt = `\n用户问题: ${userInput}`;
      // 发送用户消息和安全事件信息到后端
      axios.post('http://43.138.200.89:8080/kairos/completions', { prompt: prompt })
        .then(response => {
          const aiResponse = response.data.advice;
          const aiMessage = { type: 'ai', content: aiResponse, typedText: '', typeIndex: 0 };
          this.messages.push(aiMessage);
          this.scrollToBottom();
          this.typeNextChar(aiMessage);
        })
        .catch(error => {
          console.error('Error getting AI response:', error);
        });
      },
      selectDefaultMessage(message) {
        this.userInput = message
        this.sendMessage()
      },
      addMessage(message) {
        this.messages.push(message)
        this.saveMessages()
        this.$nextTick(() => {
          this.scrollToBottom()
        })
      },
      saveMessages() {
        localStorage.setItem('chatMessages', JSON.stringify(this.messages))
      },
      loadMessages() {
        const savedMessages = localStorage.getItem('chatMessages')
        if (savedMessages) {
          this.messages = JSON.parse(savedMessages)
        }
      },
      scrollToBottom() {
        const chatMessages = this.$refs.chatMessages
        chatMessages.scrollTop = chatMessages.scrollHeight
      }
    },
    mounted() {
      this.loadMessages()
      this.$nextTick(() => {
        this.scrollToBottom()
      })
    }
  }
  </script>
  
  <style scoped>
  .chat-container {
    position: relative;
    height: 100vh;
    padding: 20px;
    box-sizing: border-box;
    display: flex;
    flex-direction: column;
    background-color: #f0f2f5; /* 添加背景色以增强对比度 */
  }
  
  .chat-messages {
    flex-grow: 1;
    overflow-y: auto;
    border: 1px solid #e6e6e6;
    border-radius: 8px; /* 增加圆角 */
    padding: 3px;
    margin-bottom: 80px;
    background-color: white; /* 添加背景色 */
    box-shadow: 0 2px 12px 0 rgba(0, 0, 0, 0.1); /* 为整个消息区域添加阴影 */
  }
  
  .default-messages {
    position: fixed;
    bottom: 60px; /* 固定位置，使其在输入框上方 */
    left: 60%;
    transform: translateX(-50%);
    display: flex;
    justify-content: space-around;
    margin-bottom: 10px;
  }
  
  .default-messages button {
    background-color: #409EFF;
    color: white;
    border: none;
    padding: 10px 20px;
    border-radius: 8px;
    cursor: pointer;
    transition: background-color 0.3s ease;
  }
  
  .default-messages button:hover {
    background-color: #66b1ff;
  }
  
  .message {
    margin-bottom: 15px;
    display: flex;
    align-items: center; /* 使头像和消息内容垂直对齐 */
  }
  
  .message-avatar {
    width: 40px; /* 设置头像宽度 */
    height: 40px; /* 设置头像高度 */
    border-radius: 50%; /* 设置圆形头像 */
    margin-right: 10px; /* 调整头像和消息内容的间距 */
  }
  
  .user {
    justify-content: flex-end;
  }
  
  .ai {
    justify-content: flex-start;
  }
  
  .message-content {
    max-width: 70%;
    padding: 3px 4px; /* 增加内边距 */
    border-radius: 18px; /* 增加圆角 */
    word-wrap: break-word;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.15); /* 为每条消息添加阴影 */
  }
  
  .user .message-content {
    background-color: #409EFF;
    color: white;
  }
  
  .ai-message {
    background-color: white;
    color: #303133;
    margin-right: 30%;
  }
  
  .chat-input {
    position: fixed;
    left: 60%;
    bottom: 20px;
    transform: translateX(-50%);
    width: 80%;
    max-width: 600px;
    z-index: 1000;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15); /* 为输入框添加阴影 */
    border-radius: 8px; /* 增加输入框圆角 */
    overflow: hidden; /* 确保圆角效果应用到子元素 */
  }
  
  /* 确保 el-input 内部的输入框和按钮样式正确 */
  :deep(.el-input-group__append) {
    background-color: #409EFF;
    border-color: #409EFF;
    color: white;
  }
  
  :deep(.el-input__inner) {
    border-top-right-radius: 0;
    border-bottom-right-radius: 0;
    border: none; /* 移除输入框边框 */
  }
  
  /* 添加过渡效果 */
  .message-content {
    transition: all 0.3s ease;
  }
  
  .message-content:hover {
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2); /* 悬停时增强阴影效果 */
  }
  </style>
  