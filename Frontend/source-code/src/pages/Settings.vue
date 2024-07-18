<template>
    <div class="col-12 settings">
      <h2>设置页面</h2>
      <div class="form-group">
        <label for="sourceUpdateInterval">源数据更新间隔（分钟）:</label>
        <input type="number" id="sourceUpdateInterval" class="form-control" v-model="sourceUpdateInterval">
      </div>
      <div class="form-group">
        <label for="sourceSaveInterval">源数据保存时间（天）:</label>
        <input type="number" id="sourceSaveInterval" class="form-control" v-model="sourceUpdateInterval">
      </div>
      <button class="btn btn-primary" @click="saveSettings">保存设置</button>
    </div>
  </template>
  
  <script>
  export default {
    data() {
      return {
      sourceUpdateInterval:2,
      sourceSaveInterval:7,
      };
    },
    methods: {
      saveSettings() {
        // 保存设置到本地存储或发送到后端
        localStorage.setItem('sourceUpdateInterval', this.sourceUpdateInterval);
        localStorage.setItem('sourceSaveInterval', this.sourceSaveInterval);
        alert('设置已保存');
      }
    },
    mounted() {
      // 在页面加载时从本地存储中获取之前保存的设置
      const savedSourceInterval = localStorage.getItem('sourceUpdateInterval');
      const sourceSaveInterval = localStorage.getItem('sourceSaveInterval');
      if (savedSourceInterval||sourceSaveInterval) {
        this.sourceUpdateInterval = parseInt(savedSourceInterval);
        this.sourceSaveInterval = parseInt(sourceSaveInterval);
      }
    }
  };
  </script>
  
  <style scoped>
  .settings {
    margin: auto;
    padding: 20px;
    background-color: #f8f9fa;
    border-radius: 5px;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
  }
  
  .form-group {
    margin-bottom: 20px;
  }
  
  .btn-primary {
    margin-top: 10px;
  }
  </style>
  