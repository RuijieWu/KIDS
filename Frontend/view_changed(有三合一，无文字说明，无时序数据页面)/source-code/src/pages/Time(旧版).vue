<template>
    <div class="time-series-display">
      <el-card class="chart-card">
        <div slot="header">
          <span>时序数据展示</span>
        </div>
        <div ref="chartContainer" class="chart-container"></div>
      </el-card>
    </div>
  </template>
  
  <script>
  import * as echarts from 'echarts';
  
  export default {
    name: 'TimeSeriesDisplay',
    data() {
      return {
        chart: null,
        data: [],
        edges: [],
      };
    },
    mounted() {
      this.initChart();
      this.startDataSimulation();
    },
    methods: {
      initChart() {
        const chartDom = this.$refs.chartContainer;
        this.chart = echarts.init(chartDom);
        
        this.data = [
          {
            fixed: true,
            x: this.chart.getWidth() / 2,
            y: this.chart.getHeight() / 2,
            symbolSize: 20,
            id: '-1'
          }
        ];
  
        const option = {
          series: [
            {
              type: 'graph',
              layout: 'force',
              animation: false,
              data: this.data,
              force: {
                repulsion: 100,
                edgeLength: 5
              },
              edges: this.edges
            }
          ]
        };
  
        this.chart.setOption(option);
      },
      startDataSimulation() {
        setInterval(() => {
          this.data.push({
            id: this.data.length + ''
          });
          
          const source = Math.round((this.data.length - 1) * Math.random());
          const target = Math.round((this.data.length - 1) * Math.random());
          
          if (source !== target) {
            this.edges.push({
              source: source,
              target: target
            });
          }
  
          this.chart.setOption({
            series: [
              {
                roam: true,
                data: this.data,
                edges: this.edges
              }
            ]
          });
        }, 200);
      }
    },
    beforeDestroy() {
      if (this.chart) {
        this.chart.dispose();
      }
    }
  }
  </script>
  
  <style scoped>
  .time-series-display {
    padding: 20px;
  }
  
  .chart-card {
    width: 100%;
  }
  
  .chart-container {
    height: 600px;
  }
  </style>