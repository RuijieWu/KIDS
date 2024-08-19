<template>
    <div>
    <div>
      <h3>数据仓库监控</h3>
      <el-button @click="toggleChart" class="toggle-button">
        切换图表
      </el-button>
      <el-button @click="toggleChart" class="toggle-button">
        添加新的仓库
      </el-button>
    </div>
      <div ref="chartContainer" class="chart"></div>
    </div>
  </template>
  
  <script>
  import * as echarts from 'echarts';
  import 'echarts-liquidfill';
  
  export default {
    name: 'DataWarehouseVisualization',
    data() {
      return {
        warehouses: [
          { name: "主数据仓库", capacity: 1000, usedCapacity: 750 },
          { name: "历史数据仓库", capacity: 750, usedCapacity: 637 },
          { name: "分析数据仓库", capacity: 500, usedCapacity: 250 },
          { name: "备份数据仓库", capacity: 250, usedCapacity: 200 },
          { name: "临时数据仓库", capacity: 100, usedCapacity: 30 },
          { name: "归档数据仓库", capacity: 800, usedCapacity: 720 },
          { name: "实时数据仓库", capacity: 300, usedCapacity: 150 },
          { name: "测试数据仓库", capacity: 50, usedCapacity: 25 },
          { name: "开发数据仓库", capacity: 150, usedCapacity: 100 },
          { name: "灾备数据仓库", capacity: 600, usedCapacity: 300 }
        ],
        showTotal: false // 新增变量控制图表显示状态
      }
    },
    mounted() {
      this.initChart();
      this.updateChart();
      window.addEventListener('resize', this.handleResize);
    },
    beforeUnmount() {
      if (this.chart) {
        this.chart.dispose();
        this.chart = null;
      }
      window.removeEventListener('resize', this.handleResize);
    },
    computed: {
      totalCapacity() {
        return this.warehouses.reduce((total, warehouse) => total + warehouse.capacity, 0);
      },
      totalUsedCapacity() {
        return this.warehouses.reduce((total, warehouse) => total + warehouse.usedCapacity, 0);
      },
      currentData() {
        if (this.showTotal) {
          return [{
            name: '总体数据',
            value: this.totalCapacity,
            capacity: this.totalCapacity,
            usedCapacity: this.totalUsedCapacity,
            itemStyle: {
              color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [{
                offset: 0,
                color: 'rgba(0,255,255,0.3)'
              }, {
                offset: this.totalUsedCapacity / this.totalCapacity,
                color: 'rgba(0,255,255,0.8)'
              }, {
                offset: this.totalUsedCapacity / this.totalCapacity,
                color: 'rgba(0,255,255,0.1)'
              }, {
                offset: 1,
                color: 'rgba(0,255,255,0.1)'
              }])
            },
            label: {
              show: true,
              formatter: (params) => {
                return [
                  `{name|${params.name}}`,
                  `{rate|${(params.data.usedCapacity / params.data.capacity * 100).toFixed(0)}%}`
                ].join('\n');
              },
              rich: {
                name: {
                  fontSize: 14,
                  color: '#000'
                },
                rate: {
                  fontSize: 20,
                  color: '#000',
                  fontWeight: 'bold'
                }
              }
            }
          }];
        } else {
          return this.warehouses.map(warehouse => ({
            name: warehouse.name,
            value: warehouse.capacity,
            capacity: warehouse.capacity,
            usedCapacity: warehouse.usedCapacity,
            itemStyle: {
              color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [{
                offset: 0,
                color: 'rgba(0,255,255,0.3)'
              }, {
                offset: warehouse.usedCapacity / warehouse.capacity,
                color: 'rgba(0,255,255,0.8)'
              }, {
                offset: warehouse.usedCapacity / warehouse.capacity,
                color: 'rgba(0,255,255,0.1)'
              }, {
                offset: 1,
                color: 'rgba(0,255,255,0.1)'
              }])
            },
            label: {
              show: true,
              formatter: (params) => {
                return [
                  `{name|${params.name}}`,
                  `{rate|${(params.data.usedCapacity / params.data.capacity * 100).toFixed(0)}%}`
                ].join('\n');
              },
              rich: {
                name: {
                  fontSize: 14,
                  color: '#000'
                },
                rate: {
                  fontSize: 20,
                  color: '#000',
                  fontWeight: 'bold'
                }
              }
            }
          }));
        }
      }
    },
    methods: {
      initChart() {
        this.chart = echarts.init(this.$refs.chartContainer);
      },
      updateChart() {
        const option = {
          title: {
          },
          tooltip: {
            formatter: (info) => {
              const data = info.data;
              return [
                data.name,
                `总容量: ${data.capacity}`,
                `已使用: ${data.usedCapacity}`,
                `使用率: ${(data.usedCapacity / data.capacity * 100).toFixed(2)}%`
              ].join('<br>');
            }
          },
          series: [{
            type: 'treemap',
            data: this.currentData,
            levels: [{
              itemStyle: {
                borderColor: '#fff',
                borderWidth: 2,
                gapWidth: 2
              }
            }]
          }]
        };
        this.chart.setOption(option);
      },
      handleResize() {
        if (this.chart) {
          this.chart.resize();
        }
      },
      toggleChart() {
        this.showTotal = !this.showTotal;
        this.updateChart();
      }
    }
  }
  </script>
  
  <style scoped>
  .chart {
    height: 600px;
    width: 100%;
    margin-top: 20px;
  }
  .toggle-button:hover {
    background-color: #0056b3;
  }
  </style>
  