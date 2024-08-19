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
      <div class="col-12 loading" v-loading="loading" element-loading-text="加载中...">
        <source-table :sources="filteredSources" :title="EventTable.title"></source-table>
      </div>
    </div>
  </template>
  
  <script>
  import SourceTable from '@/components/SourceTable.vue';
  import axios from 'axios';
  export default {
    components: {
      SourceTable,
    },
    data() {
      return {
        loading:false,
        searchText: '',
        EventTable: {
          title: "警告信息",
          data:[],
          ids: [],
        },
      };
    },
    computed: {
      filteredSources() {
        if (!this.searchText) {
          return this.EventTable.data;
        }
        const searchLower = this.searchText.toLowerCase();
        return this.EventTable.data.filter(source => 
          Object.values(source).some(value => 
            String(value).toLowerCase().includes(searchLower)
          )
        );
      },
    },
    mounted() {
    this.fetchEventData();
    const sourceUpdateIntervalMinutes = parseInt(localStorage.getItem('sourceUpdateInterval')) || 2;
    const sourceUpdateIntervalMilliseconds = sourceUpdateIntervalMinutes * 60 * 1000;
  },
    methods: {
      
      searchSources() {
        console.log('Searching for:', this.searchText);
      },


      async fetchEventData() {
        this.loading = true;
  try {
    const response = await axios.get(`http://43.138.200.89:8080/kairos/graph/info`, {
      params: {
        start_time: '2018-04-01 00:00:00',
        end_time: '2018-04-12 00:00:00',
      },
      headers: {
        'content-type': 'application/json;charset=utf-8',
        // 根据需要添加授权信息
      }
    });

    if (response.data && response.data.data) {
      this.EventTable.data = response.data.data.map((item, index) => {
        // 从文件名中提取时间段
        const timeInfo = this.extractTimeInfo(item.file_name);

        // 随机生成危害级别，实际应用中应根据具体逻辑决定
        const dangerLevels = ['low', 'medium', 'high'];
        let randomDangerLevel = dangerLevels[Math.floor(Math.random() * dangerLevels.length)];
        if(item.dangerous_action_count == 0&&item.dangerous_subject_count == 0 &&item.dangerous_object_count ==0) randomDangerLevel = 'low';
        else if(item.dangerous_action_count != 0||item.dangerous_subject_count != 0 ||item.dangerous_object_count != 0&&
        item.dangerous_action_count * item.dangerous_subject_count * item.dangerous_object_count == 0) randomDangerLevel = 'medium';
        else if (item.dangerous_action_count != 0&&item.dangerous_subject_count != 0 &&item.dangerous_object_count != 0) randomDangerLevel = 'high';
        return {
          ID: index + 1,
          开始时间: timeInfo.startTime,
          结束时间: timeInfo.endTime,
          可疑行为数: item.anomalous_action_count,
          可疑攻击方数: item.anomalous_subject_count,
          可疑被攻击方数: item.anomalous_object_count,
          危险行为数: item.dangerous_action_count,
          危险攻击方数: item.dangerous_subject_count,
          危险被攻击方数: item.dangerous_object_count,
          危害级别: randomDangerLevel,
          文件名: this.extractFileName(item.file_name),
        };
      });
      // 更新总数
      this.EventTable.total = response.data.total;
    }
  } catch (error) {
    console.error('获取警告数据时出错:', error);
    // 这里可以添加错误处理逻辑，比如显示一个错误消息给用户
  }finally{
    this.loading = false;
  }
},
extractFileName(fileName) {
  // 将未知字符替换为冒号
  const cleanFileName = fileName.replace(/[^0-9\-: ~._^a-z]/g, ':');
  const timeMatch = cleanFileName.match(/(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d+~\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d+)(.+)/);
  if (timeMatch) {
    return timeMatch[1] + timeMatch[2];
  }
  return cleanFileName;
},
extractTimeInfo(fileName) {
  const timeMatch = fileName.match(/(\d{4}-\d{2}-\d{2} \d{2}.?\d{2}.?\d{2})\.(\d+)~(\d{4}-\d{2}-\d{2} \d{2}.?\d{2}.?\d{2})\.(\d+)/);
  if (timeMatch) {
  const formatTime = (dateString) => {
    const cleanDateString = dateString.replace(/[^0-9 -]/g, '');
    const [datePart, timePart] = cleanDateString.split(' ');
    const [year, month, day] = datePart.split('-').map(Number);
    const hour = parseInt(timePart.substr(0, 2));
    const minute = parseInt(timePart.substr(2, 2));
    const second = parseInt(timePart.substr(4, 2));
    const date = new Date(Date.UTC(year, month - 1, day, hour, minute, second));
    return `${year}-${String(month).padStart(2, '0')}-${String(day).padStart(2, '0')} ${String(hour).padStart(2, '0')}:${String(minute).padStart(2, '0')}:${String(second).padStart(2, '0')}`;
  };
  return {
    startTime: formatTime(timeMatch[1]),
    endTime: formatTime(timeMatch[3])
  };
}
  return { startTime: '未知', endTime: '未知' };
}
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
  .loading{
    height: 100%;
  }
  </style>