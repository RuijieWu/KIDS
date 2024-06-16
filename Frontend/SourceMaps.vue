<template>
    <div class="row">
      <div class="col-12">
        <div class="table-header">
          <div class="search-bar">
            <input v-model="searchText" type="text" placeholder="搜索..." class="form-control" />
            <p-button class="search-btn ti-search"></p-button>
          </div>
        </div>
      </div>
      <div class="col-12">
        <source-table :sources="filteredSources" :title="AlterTable.title"></source-table>
      </div>
    </div>
  </template>
  
  <script>
  import SourceTable from '@/components/SourceTable.vue';
  
  export default {
    components: {
      SourceTable,
    },
    data() {
      return {
        searchText: '',
        AlterTable: {
          title: "警告信息",
          data: [
            {ID:"1", 时间:"2024-6-15", 主机:"localhost", 主机名称:"localhost", 技术:"sql injection", 策略:"static analysis", 描述:"Detected potential SQL injection vulnerability", 危害级别:"high"},
            {ID:"2", 时间:"2024-6-16", 主机:"192.168.1.5", 主机名称:"webserver", 技术:"xss", 策略:"dynamic analysis", 描述:"Cross-site scripting (XSS) attempt identified", 危害级别:"medium"},
            {ID:"3", 时间:"2024-6-17", 主机:"192.168.1.10", 主机名称:"fileserver", 技术:"directory traversal", 策略:"log analysis", 描述:"Unusual file access pattern detected", 危害级别:"low"},
            {ID:"4", 时间:"2024-6-17", 主机:"192.168.1.10", 主机名称:"fileserver", 技术:"directory traversal", 策略:"log analysis", 描述:"Unusual file access pattern detected", 危害级别:"low"},
            {ID:"5", 时间:"2024-6-17", 主机:"192.168.1.10", 主机名称:"fileserver", 技术:"directory traversal", 策略:"log analysis", 描述:"Unusual file access pattern detected", 危害级别:"low"},

          ],
        },
      };
    },
    computed: {
      filteredSources() {
        if (!this.searchText) {
          return this.AlterTable.data;
        }
        const searchLower = this.searchText.toLowerCase();
        return this.AlterTable.data.filter(source => 
          Object.values(source).some(value => 
            String(value).toLowerCase().includes(searchLower)
          )
        );
      },
    },
    methods: {
      searchSources() {
        // 这里可以添加额外的搜索逻辑，如果需要的话
        console.log('Searching for:', this.searchText);
      },
      fetchAlertData() {
        // 在实际应用中，这里应该是一个 API 调用
        // 现在我们just使用模拟数据
        console.log('Fetching alert data...');
        // 如果你后续添加了实际的API调用，可以在这里实现
      },
    },
    created() {
      this.fetchAlertData();
    },
  };
  </script>
  
  <style scoped>
  .table-header {
    display: flex;
    justify-content: flex-end;
    align-items: center;
    margin-bottom: 20px;
  }
  
  .search-bar {
    display: flex;
    align-items: center;
  }
  
  .search-btn {
    margin-left: 8px;
  }
  
  .form-control {
    padding: 8px 12px;
    border: 1px solid #ced4da;
    border-radius: 4px;
    font-size: 14px;
    width: 300px;
  }
  </style>