
<template>
    <!-- 你的现有模板 -->
  </template>
  
  <script>
  import axios from 'axios';
  
  export default {
    data() {
      return {
        chartData: [],
        loading: false,
        error: null
      };
    },
    methods: {
      async fetchChartData() {
        try {
            const today = new Date();
            today.setHours(0, 0, 0, 0); // 将时间设为当天的开始
            for (let i = 0; i < 7; i++) {
                const day = new Date(today);
                const formattedDate = (`0${day.getMonth() + 1}`).slice(-2) + '-' + (`0${day.getDate()}`).slice(-2); // 格式化日期为 MM-DD
                this.usersChart.labels[6-i] = formattedDate;
          }
            const formattedDates = []; // 用于存储格式化后的日期时间
            for (let i = 0; i < this.usersChart.labels.length; i++) {
                const mmDd = this.usersChart.labels[i]; // 取出 MM-DD 格式的日期
                const [month, day] = mmDd.split('-'); // 分割月份和日期
                const now = new Date();
                const year = now.getFullYear();
                const formattedDate = `${year}-${(`0${month}`).slice(-2)}-${(`0${day}`).slice(-2)} 00:00:00`;
                formattedDates.push(formattedDate);
            }   
            const now = new Date();
            const year = now.getFullYear();
            const currentMonth = (`0${now.getMonth() + 1}`).slice(-2); // 月份从 0 开始，所以加 1
            const currentDay = (`0${now.getDate()}`).slice(-2);
            const hours = (`0${now.getHours()}`).slice(-2);
            const minutes = (`0${now.getMinutes()}`).slice(-2);
            const seconds = (`0${now.getSeconds()}`).slice(-2);
            const currentFormattedDate = `${year}-${currentMonth}-${currentDay} ${hours}:${minutes}:${seconds}`;
            formattedDates.push(currentFormattedDate);
            for(let i = 0; i < 7; i++){
                const attackEndTime = formattedDates[7-i]
                const attackStartTime = formattedDates[6-i]
            const response = await axios.get('https://ecloud.10086.cn/api/safebox-asp/csproxy/v2/alarmCenter/alarmList', {
            params: {
              page: 1,
              pageSize: 1,
              alarmTypes: [11, 10, 8, 14, 15, 16, 13, 9, 12],
              attackStartTime:attackStartTime,
              attackEndTime:attackEndTime,
            },
            headers: {
              'Accept-Encoding': 'application/json'
              // 可能需要添加授权信息，如 token
              // 'Authorization': 'Bearer your_token_here'
            }
          });
          const alarms = response.data.body.content;
          const total = response.data.total;
          this.usersChart.data[6-i] = total;
        }
  
            
            const pastSevenDays = [];
            this.chartData = pastSevenDays.reverse();
        } catch (error) {
            this.error = '获取图表数据失败';
        } finally {
            this.loading = false;
        }
      },
      getTimeRangeParams(days) {
        const end = new Date();
        const start = new Date();
        start.setDate(end.getDate() - days);
        const formatDateTime = date => `${date.toISOString().split('T')[0]} 00:00:00`;
        return {
          attackStartTime: formatDateTime(start),
          attackEndTime: formatDateTime(end)
        };
      }
    },
    created() {
      this.fetchChartData();
    }
  };
  </script>
  
  <style>
  /* 你的现有样式 */
  </style>
  