<template>
  <div id="app" v-loading="loading" element-loading-text="加载中...">
    <el-card class="box-card">
      <div slot="header" class="clearfix">
        <span>数据仓库监控</span>
      </div>
      <el-row :gutter="20">
        <el-col :span="12">
          <div id="totalUsageChart" style="width: 100%; height: 300px;"></div>
        </el-col>
        <el-col :span="12">
          <img
            style="width: 100%; max-height: 200px; object-fit: contain; margin-bottom: 20px;"
            src="../assets/img/warehouse.jpg"
            fit="contain"
            :preview-src-list="[dataWarehouseStructureImage]">
          </img>
          <h4>数据仓库实现技术</h4>
          <ul>
            <li>分布式集群架构的多层数据仓库</li>
            <li>数据仓库分为数据采集层、数据存储与分析层和数据共享层三层。数据流在这三层之中逐级传递，最终为应用层提供数据支持。</li>
            <li>多层数据仓库结构整合了不同技术的优势特色，可以实现海量数据存储和动态扩容</li>
          </ul>
        </el-col>
      </el-row>
      <el-row :gutter="20" style="margin-top: 20px;">
        <el-col :span="12" v-for="(warehouse, index) in warehousesData" :key="index">
          <el-card class="warehouse-card">
            <div slot="header" class="clearfix">
              <span>
                {{ warehouse.storage_name }}
                <el-tooltip 
                  :content="getStorageDescription(warehouse.storage_name)" 
                  placement="top-start"
                >
                  <i class="el-icon-question"></i>
                </el-tooltip>
              </span>
            </div>
            <div class="warehouse-info">
              <p>已使用容量: {{ warehouse.used_space }}</p>
              <p>总容量: {{ warehouse.total_space }}</p>
              <el-progress 
                :percentage="calculatePercentage(warehouse)"
                :color="determineColor(warehouse)">
              </el-progress>
              <el-collapse style="margin-top: 10px;">
                <el-collapse-item title="查看子节点详情">
                  <el-table :data="warehouse.nodes" style="width: 100%">
                    <el-table-column prop="node_name" label="节点名称"></el-table-column>
                    <el-table-column prop="used_space" label="已使用容量"></el-table-column>
                    <el-table-column prop="total_space" label="总容量"></el-table-column>
                    <el-table-column label="使用率">
                      <template slot-scope="scope">
                        <el-progress 
                          :percentage="calculateNodePercentage(scope.row)"
                          :color="determineColor(scope.row)">
                        </el-progress>
                      </template>
                    </el-table-column>
                    <el-table-column prop="transactions" label="事务提交"></el-table-column>
                    <el-table-column prop="rollbacks" label="事务回滚"></el-table-column>
                    <el-table-column prop="blks_read" label="块读取"></el-table-column>
                    <el-table-column prop="blks_hit" label="块命中"></el-table-column>
                  </el-table>
                </el-collapse-item>
              </el-collapse>
            </div>
          </el-card>
        </el-col>
      </el-row>
    </el-card>
  </div>
</template>

<script>
import * as echarts from 'echarts'
import axios from 'axios'

export default {
  name: 'App',
  data() {
    return {
      warehousesData: [],
      loading:false,
      dataWarehouseStructureImage: '../assets/img/warehouse.jpg',
    }
  },
  mounted() {
    this.fetchData()
  },
  methods: {
    getStorageDescription(storageName) {
    if (storageName === 'HDFS Storage') {
      return 'HDFS (Hadoop Distributed File System) 是数据存储与分析层的数据仓库，HDFS存储数据采集模块Agent采集的数据，并且直接将数据交付给KIDS引擎。'
    } else if (storageName === 'Postgres Storage') {
      return 'PostgreSQL 是数据共享层的数据仓库，PostgreSQL获取规则引擎预检的数据和KIDS引擎交付的数据提供给应用层即接口和前端使用。'
    }
    return '数据存储系统'
  },
    async fetchData() {
      this.loading = true;
      try {
        const response = await axios.get('http://43.138.200.89:8080/data/stats')
        this.warehousesData = response.data.warehouses
        this.initTotalUsageChart()
      } catch (error) {
        console.error('Error fetching data:', error)
      }finally{
        this.loading = false;
      }
    },
    calculatePercentage(warehouse) {
      const used = this.parseSize(warehouse.used_space);
      const total = this.parseSize(warehouse.total_space);
      console.log("使用量",used,total,warehouse.used_space,warehouse.total_space)
      console.log("使用率",Math.round((used / total) * 100))
      return Math.round((used / total) * 100) || 0;
    },

    calculateNodePercentage(node) {
      const used = this.parseSize(node.used_space);
      const total = this.parseSize(node.total_space);
      return Math.round((used / total) * 100) || 0;
    },

    parseSize(sizeString) {
      const match = sizeString.match(/(\d+(\.\d+)?)\s*(\w+)/);
      if (!match) return 0;
      const [, size, , unit] = match;
      const multipliers = { B: 1, kB: 1024, MB: 1024 * 1024, GB: 1024 * 1024 * 1024, TB: 1024 ** 4, PB: 1024 ** 5 };
      return parseFloat(size) * (multipliers[unit.toUpperCase()] || 1);
    },

    determineColor(item) {
      const percentage = this.calculatePercentage(item)
      if (percentage >= 90) {
        return '#FF4949'  // 红色
      } else if (percentage >= 70) {
        return '#F7BA2A'  // 橙色
      } else {
        return '#13CE66'  // 绿色
      }
    },
    initTotalUsageChart() {
      const chartDom = document.getElementById('totalUsageChart')
      const myChart = echarts.init(chartDom)
      const totalUsed = this.warehousesData.reduce((sum, warehouse) => sum + this.parseSize(warehouse.used_space), 0)
      const totalCapacity = this.warehousesData.reduce((sum, warehouse) => sum + this.parseSize(warehouse.total_space), 0)
      
      const option = {
        title: {
          text: '总体使用情况',
          left: 'center'
        },
        tooltip: {
          trigger: 'item',
          formatter: '{a} <br/>{b}: {c} GB ({d}%)'
        },
        legend: {
          orient: 'vertical',
          left: 'left'
        },
        series: [
          {
            name: '容量',
            type: 'pie',
            radius: '50%',
            data: [
              { value: totalUsed / (1024 * 1024 * 1024), name: '已使用' },
              { value: (totalCapacity - totalUsed) / (1024 * 1024 * 1024), name: '未使用' }
            ],
            emphasis: {
              itemStyle: {
                shadowBlur: 10,
                shadowOffsetX: 0,
                shadowColor: 'rgba(0, 0, 0, 0.5)'
              }
            }
          }
        ]
      }
      
      myChart.setOption(option)
    }
  }
}
</script>

<style>
#app {
  font-family: Avenir, Helvetica, Arial, sans-serif;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  color: #2c3e50;
  margin-top: 60px;
}
.warehouse-card {
  margin-bottom: 20px;
}
.warehouse-info {
  text-align: left;
}
/* 新增样式 */
ul {
  text-align: left;
  padding-left: 20px;
}
li {
  margin-bottom: 10px;
}
</style>