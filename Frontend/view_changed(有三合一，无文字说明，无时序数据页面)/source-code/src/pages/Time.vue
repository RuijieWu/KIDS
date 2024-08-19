<template>
    <div ref="chart" style="width: 100%; height: 600px;"></div>
  </template>
  
  <script>
  import axios from 'axios';
  import * as echarts from 'echarts';
  
  export default {
    data() {
      return {
        chart: null,
        chartData: [],
      };
    },
    mounted() {
      this.initChart();
      this.fetchData();
    },
    methods: {
      initChart() {
        this.chart = echarts.init(this.$refs.chart);
      },
      async fetchData() {
        try {
          const response = await axios.get('http://43.138.200.89:8080/kairos/graph/info',{
      params: {
        start_time: '2018-04-01 00:00:00',
        end_time: '2018-04-12 00:00:00',
      },
      headers: {
        'content-type': 'application/json;charset=utf-8',
      }
    });
          if (response.data && response.data.data) {
            let eventsByDate = {};
            response.data.data.forEach(item => {
              const date = this.extractDate(item.file_name);
              if (!eventsByDate[date]) {
                eventsByDate[date] = 0;
              }
              eventsByDate[date]++;
            });
  
            let cumulativeSum = 0;
            this.chartData = Object.entries(eventsByDate)
              .sort(([a], [b]) => new Date(a) - new Date(b))
              .map(([date, count]) => {
                cumulativeSum += count;
                return [date, cumulativeSum];
              });
  
            this.updateChart();
          }
        } catch (error) {
          console.error('Error fetching data:', error);
        }
      },
      extractDate(fileName) {
        const match = fileName.match(/(\d{4}-\d{2}-\d{2})/);
        return match ? match[1] : '';
      },
      updateChart() {
        const option = {
          animationDuration: 10000,
          title: {
            text: '累积安全事件数 (2018-04-01 到 2018-04-12)'
          },
          tooltip: {
            trigger: 'axis',
            formatter: function(params) {
              return `日期: ${params[0].data[0]}<br/>累积事件数: ${params[0].data[1]}`;
            }
          },
          xAxis: {
            type: 'time',
            min: '2018-04-01',
            max: '2018-04-12',
            nameLocation: 'middle',
            axisLabel: {
              formatter: '{yyyy}-{MM}-{dd}'
            }
          },
          yAxis: {
            name: '累积事件数',
            type: 'value',
            minInterval: 1
          },
          series: [{
            type: 'line',
            data: this.chartData,
            showSymbol: true,
            endLabel: {
              show: true,
              formatter: function (params) {
                return params.data[1];
              }
            },
            labelLayout: {
              moveOverlap: 'shiftY'
            },
            emphasis: {
              focus: 'series'
            }
          }]
        };
        this.chart.setOption(option);
      }
    }
  };
  </script>