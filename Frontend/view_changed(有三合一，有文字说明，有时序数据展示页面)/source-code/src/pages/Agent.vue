<template>
  <div class="agent-monitor">
    <h3>Agent 状态监控（部署在不同主机用于采集数据）</h3>
    
    <el-row :gutter="20">
      <div class="source-child" v-loading="loading" element-loading-text="加载中...">
      <el-col :span="12">
        <el-card class="box-card">
          <div slot="header" class="clearfix">
            <span>Agent 状态概览</span>
          </div>
          <el-row :gutter="20">
            <el-col :span="12">
              <el-card shadow="hover" class="status-card online">
                <div class="status-title">在线</div>
                <div class="status-count">{{ onlineCount }}</div>
              </el-card>
            </el-col>
            <el-col :span="12">
              <el-card shadow="hover" class="status-card offline">
                <div class="status-title">离线</div>
                <div class="status-count">{{ offlineCount }}</div>
              </el-card>
            </el-col>
          </el-row>
        </el-card>
      </el-col>
    </div>
    <div class="source-child" v-loading="loading" element-loading-text="加载中...">
      <el-col :span="12">
        <el-card class="box-card">
          <div slot="header" class="clearfix">
            <span>Agent 状态分布</span>
          </div>
          <div id="chart" style="height: 300px;"></div>
        </el-card>
      </el-col>
    </div>
    </el-row>
    <div class="source-child" v-loading="loading" element-loading-text="加载中...">
    <el-card class="box-card" style="margin-top: 20px;">
      <div slot="header" class="clearfix">
        <span>Agent 详细信息</span>
        <el-input
          v-model="search"
          placeholder="搜索"
          style="width: 200px; float: right;"
        ></el-input>
      </div>
      <el-table
        :data="paginatedAgents"
        style="width: 100%"
      >
        <el-table-column prop="hostname" label="主机名"></el-table-column>
        <el-table-column prop="host_ip" label="IP地址"></el-table-column>
        <el-table-column prop="status" label="状态">
          <template slot-scope="scope">
            <el-tag
              :type="scope.row.status === 'alive' ? 'success' : 'danger'"
            >
              {{ scope.row.status === 'alive' ? '在线' : '离线' }}
            </el-tag>
          </template>
        </el-table-column>
        <el-table-column prop="uptime" label="运行时间"></el-table-column>
        <el-table-column prop="paths" label="路径">
          <template slot-scope="scope">
            <el-button
              size="mini"
              @click="showPaths(scope.row.paths)"
            >
              查看路径
            </el-button>
          </template>
        </el-table-column>
      </el-table>
      <el-pagination
        @size-change="handleSizeChange"
        @current-change="handleCurrentChange"
        :current-page="currentPage"
        :page-sizes="[5, 10, 20, 50]"
        :page-size="pageSize"
        layout="total, sizes, prev, pager, next, jumper"
        :total="filteredAgents.length">
      </el-pagination>
    </el-card>
</div>
    <el-dialog 
      title="路径列表" 
      :visible.sync="dialogVisible"
      :modal-append-to-body="false"
      custom-class="path-dialog">
      <el-list>
        <el-list-item v-for="(path, index) in dialogPaths" :key="index">
          <pre>{{ path }}</pre>
        </el-list-item>
      </el-list>
    </el-dialog>

  </div>
</template>

<script>
import axios from 'axios';
import * as echarts from 'echarts';

export default {
  data() {
    return {
      loading:false,
      agents: [],
      search: '',
      dialogVisible: false,
      dialogPaths: [],
      currentPage: 1,
      pageSize: 5,
    };
  },
  computed: {
    onlineCount() {
      return this.agents.filter(a => a.status === 'alive').length;
    },
    offlineCount() {
      return this.agents.length - this.onlineCount;
    },
    filteredAgents() {
      return this.agents.filter(agent =>
        Object.values(agent).some(value =>
          value.toString().toLowerCase().includes(this.search.toLowerCase())
        )
      );
    },
    paginatedAgents() {
      const start = (this.currentPage - 1) * this.pageSize;
      const end = start + this.pageSize;
      return this.filteredAgents.slice(start, end);
    }
  },
  methods: {
    async fetchData() {
  this.loading = true;
  try {
    const response = await axios.get('http://43.138.200.89:8080/data/agent-info');
    this.agents = response.data.map(agent => {
      if (agent.status === 'offline') {
        return {
          ...agent,
          hostname: agent.hostname || 'NONE',
          uptime: agent.uptime || 'NONE'
        };
      }
      return agent;
    });
    this.renderChart();
  } catch (error) {
    console.error('Error fetching agent data:', error);
    this.$message.error('获取代理数据失败，请稍后重试');
  } finally {
    this.loading = false;
  }
},
    renderChart() {
      // 确保在组件挂载后再初始化图表
      this.$nextTick(() => {
        const chart = echarts.init(document.getElementById('chart'));
        const option = {
          tooltip: {
            trigger: 'item'
          },
          legend: {
            top: '5%',
            left: 'center'
          },
          series: [
            {
              name: 'Agent 状态',
              type: 'pie',
              radius: ['40%', '70%'],
              avoidLabelOverlap: false,
              itemStyle: {
                borderRadius: 10,
                borderColor: '#fff',
                borderWidth: 2
              },
              label: {
                show: false,
                position: 'center'
              },
              emphasis: {
                label: {
                  show: true,
                  fontSize: '40',
                  fontWeight: 'bold'
                }
              },
              labelLine: {
                show: false
              },
              data: [
                { value: this.onlineCount, name: '在线' },
                { value: this.offlineCount, name: '离线' }
              ]
            }
          ]
        };

        chart.setOption(option);
      });
    },
    showPaths(paths) {
      this.dialogPaths = paths;
      this.dialogVisible = true;
    },
    handleSizeChange(val) {
      this.pageSize = val;
      this.currentPage = 1;
    },
    handleCurrentChange(val) {
      this.currentPage = val;
    },
    showPaths(paths) {
      this.dialogPaths = paths;
      this.dialogVisible = true;
    }
  },
  mounted() {
    this.fetchData();
  }
};
</script>

<style scoped>
.agent-monitor {
  padding: 20px;
}
.status-card {
  text-align: center;
  padding: 20px;
}
.status-card.online {
  background-color: #f0f9eb;
  color: #67c23a;
}
.status-card.offline {
  background-color: #fef0f0;
  color: #f56c6c;
}
.status-title {
  font-size: 18px;
  margin-bottom: 10px;
}
.status-count {
  font-size: 24px;
  font-weight: bold;
}
.path-dialog .el-dialog__body {
  max-height: 300px;
  overflow-y: auto;
}
.path-dialog pre {
  white-space: pre-wrap;
  word-wrap: break-word;
}
</style>