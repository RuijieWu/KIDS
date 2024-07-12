<template>
    <div>
      <button @click="showTextArea = !showTextArea">{{ showTextArea ? '收起' : '添加文本' }}</button>
      <div v-if="showTextArea">
        <textarea v-model="inputText" rows="5" placeholder="请输入文本..."></textarea>
        <button @click="submitText">提交</button>
      </div>
      <div v-if="submittedText">
        <h3>已提交的文本:</h3>
        <p>{{ submittedText }}</p>
      </div>
    </div>
  </template>
  
  <script>
  import { ref } from 'vue'
  
  export default {
    setup() {
      const showTextArea = ref(false)
      const inputText = ref('')
      const submittedText = ref('')
  
      const submitText = () => {
        submittedText.value = inputText.value
        inputText.value = ''
      }
  
      return {
        showTextArea,
        inputText,
        submittedText,
        submitText
      }
    }
  }
  </script>