<!-- SourceChild.vue -->
<template>
    <div class="source-child">
      <div class="row">
        <div class="col-md-12">
          <div class="d-flex justify-content-between align-items-center mb-3">
            <h4>{{ source.ID }}. {{ source.主机名称 }}</h4>
            <div>
              <span :class="['badge', getBadgeClass(source.危害级别)]">{{ source.危害级别 }}</span>
            </div>
          </div>
          <!-- 其他告警信息 -->
          <div class="mb-3">
            <strong>开始时间：</strong> {{ source.开始时间 }}
          </div>
          <div class="mb-3">
            <strong>结束时间：</strong> {{ source.结束时间 }}
          </div>
          <div class="mb-3">
            <strong>可疑行为数：</strong> {{ source.可疑行为数 }}
          </div>
          <div class="mb-3">
            <strong>可疑攻击方数：</strong> {{ source.可疑攻击方数 }}
          </div>
          <div class="mb-3">
            <strong>可疑被攻击方数：</strong> {{ source.可疑被攻击方数 }}
          </div>
          <div class="mb-3">
            <strong>危险行为数：</strong> {{ source.危险行为数 }}
          </div>
          <div class="mb-3">
            <strong>危险攻击方数：</strong> {{ source.危险攻击方数 }}
          </div>
          <div class="mb-3">
            <strong>危险被攻击方数：</strong> {{ source.危险被攻击方数 }}
          </div>
          
          <div class="row">
          <button class="btn btn-primary mr-3 ml-2" @click="showImageModal">查看相关图片</button>
          <button class="btn btn-primary" @click="showReport">查看安全报告</button>
          <button class="btn btn-primary ml-3" @click="showTable">查看行为表格</button>
        </div>
        </div>
      </div>
  
      <!-- 全屏图片模态框 -->
      <div v-if="showModalImage" class="modal-overlay" @click="closeImageModal">
  <div class="modal-content" @click.stop>
    <button class="close-btn" @click="closeImageModal">&times;</button>
    <img :src="currentImage" alt="Related Image" class="img-fluid" v-if="currentImage">
    <div v-else>没有可用的图片</div>
    <span style="display: block; text-align: center;">{{source.文件名}}</span>
    <div class="modal-navigation">
      <button class="btn btn-secondary" @click="prevImage" :disabled="currentImageIndex === 0">上一张</button>
      <button class="btn btn-secondary" @click="nextImage" :disabled="currentImageIndex === images.length - 1">下一张</button>
    </div>
  </div>
</div>

      <div v-if="showModalReport" class="modal-overlay" @click="closeReportModal">
  <div class="modal-content" @click.stop v-if="showModalReport">
    <button class="close-btn" @click="closeReportModal">&times;</button>
    <div id="echarts-container" style="width: 100%; height: 600px;"></div>
    <div class="row">
    <button @click="toggleEdgeStyle" class="btn btn-primary mr-3 ml-2">
      {{ isStraightLine ? '曲线显示' : '直线显示' }}
    </button>
    <button class="btn btn-primary mr-3 ml-2" @click="showNodeFilter">筛选节点</button>
  </div>
  </div>
</div>

 <!-- 节点筛选模态框 -->
 <div v-if="showNodeFilterModal" class="modal-overlay" @click="closeNodeFilterModal">
      <div class="modal-content" @click.stop>
        <button class="close-btn" @click="closeNodeFilterModal">&times;</button>
        <h3>节点筛选</h3>
        <div class="d-flex justify-content-end">
        <el-transfer
          filterable
          :filter-method="filterNodes"
          filter-placeholder="请输入节点名称"
          v-model="visibleNodes"
          :data="allNodes"
          :titles="['隐藏', '显示']"
        >
        </el-transfer>
      </div>
        <div class="d-flex justify-content-end">
        <button @click="applyNodeFilter" class="btn btn-primary mt-3">应用筛选</button>
      </div>
      </div>
    </div>

<div v-if="showNodeModal" class="modal-overlay" @click="closeNodeModal">
      <div class="modal-content" @click.stop>
        <button class="close-btn" @click="closeNodeModal">&times;</button>
        <h3>节点详情</h3>
        <p><strong>名称:</strong> {{ selectedNode.name }}</p>
        <p><strong>类型:</strong> {{ selectedNode.type }}</p>
        <button @click="hideNode" class="hide-node-btn col-2">隐藏此节点</button>
      </div>
    </div>

    <!-- 边详情模态框 -->
    <div v-if="showEdgeModal" class="modal-overlay" @click="closeEdgeModal">
      <div class="modal-content" @click.stop>
        <button class="close-btn" @click="closeEdgeModal">&times;</button>
        <h3>边详情</h3>
        <div v-for="(action, index) in selectedEdge.actions" :key="index">
          <p><strong>行为:</strong> {{ action.Action }}</p>
          <p><strong>时间:</strong> {{ formatTime(action.Time) }}</p>
          <hr v-if="index < selectedEdge.actions.length - 1">
        </div>
      </div>
    </div>

    <div v-if="showTables" class="modal-overlay" @click="closeTableModal">
  <div class="modal-content" @click.stop v-if="showTables">
    <button class="close-btn" @click="closeTableModal">&times;</button>
    <div class="col-12">
      <div class="col-12">
    <div class="table-header row">
    <div class="table-selector">
      <label for="tableSelect">选择表格:</label>
      <select id="tableSelect" v-model="selectedTable" class="form-select">
        <option value="AlterTable">可疑行为列表</option>
        <option value="DangerTable">危险行为列表</option>
      </select>
      </div>
      <div class="search-bar" >
      <input v-model="searchText" type="text" placeholder="搜索..." class="form-control" />
      <p-button class="search-btn ti-search"></p-button>
    </div>
    </div>
    </div>
      <div class="col-12">
      <card :title="CurrentTable.title" :subTitle="CurrentTable.subTitle" v-if="CurrentTable">
        <div slot="raw-content" class="table">
          <paper-table 
            :data="filteredData" 
            :columns="CurrentTable.columns">
          </paper-table>
        </div>
      </card>
      <div class="page_button">
        <p-button type="info" round @click.native="handlePrevPage" >上一页</p-button>
      <span>第 {{ currentPage }} 页 / 共 {{ totalPages }} 页</span>
      <p-button type="info" round @click.native="handleNextPage" >下一页</p-button>
    </div>
    </div>
    </div>
  </div>
</div>
    </div>
  </template>
  
  <script>
  import * as echarts from 'echarts';
  import { PaperTable } from "@/components";
  import axios from 'axios';
  const AlterTableColumns = ["时间","主体类型","主体名称","行为","客体类型","客体名称"]
  const DangerTableColumns = ["时间","主体类型","主体名称","行为","客体类型","客体名称"]
  export default {
    components: {
      PaperTable
  },
    name: 'SourceChild',
    props: {
      source: {
        type: Object,
        required: true
      }
    },
    data() {
      return {
        showTables:false,
        showModalImage: false,
        showModalReport:false,
        showNodeModal: false,
        showEdgeModal: false,
        showNodeFilterModal:false,
        isStraightLine: false,
        selectedNode: null,
        selectedEdge: null,
        currentImageIndex: 0,
        reportData: null,
        networkData: null,
        chart: null,
        hiddenNodes: [],
        allNodes: [],
        visibleNodes: [],
        allFilteredData: [],
        selectedTable: 'DangerTable',
        searchText: '',
        AlterTable: {
          title: "可疑行为列表",
          subTitle: "",
          columns: [...AlterTableColumns],
          data: [],
          options:{
            pageSize: 10, 
            currentPage: 1, 
        }
      },
        DangerTable:{
          title: "危险行为列表",
          subTitle: "",
          columns: [...DangerTableColumns],
          data: [],
          options:{
            pageSize: 10, 
            currentPage: 1, 
        }
      }
    }},
    mounted() {
      window.addEventListener('resize', this.resizeChart);
      this.fetchReports();
    },

    beforeDestroy() {
    if (this.chart) {
      this.chart.dispose();
    }
    window.removeEventListener('resize', this.resizeChart);
  },
    computed: {
  images() {
    return this.source.图片内容 || [];
  },
  currentImage() {
    const image = this.images;
    return image ? `data:image/png;base64,${image}` : null;
  },
  CurrentTable() {
    return this[this.selectedTable];
  },
  currentPage() {
      return this.CurrentTable.options.currentPage ;
    },
    // 每页条目数
  pageSize() {
      return this.CurrentTable.options.pageSize || 6;
    },
    // 总条目数
  totalItems() {
      return this.allFilteredData.length || 0;
    },
    // 总页数
  totalPages() {
      return Math.ceil(this.totalItems / this.pageSize)|| 1;
    },
    // 当前页数据
  currentPageData() {
      const startIndex = (this.currentPage - 1) * this.pageSize;
      const endIndex = startIndex + this.pageSize;
      return this.CurrentTable.data.slice(startIndex, endIndex);
    },
  filteredData() {
    const startIndex = (this.currentPage - 1) * this.pageSize;
    const endIndex = startIndex + this.pageSize;
    return this.allFilteredData.slice(startIndex, endIndex);
  },
    },

    watch: {
  searchText(newValue) {
    this.updateAllFilteredData();
  },
  CurrentTable() {
    this.updateAllFilteredData();
  },
},


    methods: {

  updateAllFilteredData() {
    const searchText = this.searchText.trim().toLowerCase();
    if (!searchText) {
      this.allFilteredData = this.CurrentTable.data;
    } else {
      this.allFilteredData = this.CurrentTable.data.filter(item => {
        return Object.values(item).some(value => {
          return String(value).toLowerCase().includes(searchText);
        });
      });
    }
  },
  handlePrevPage() {
      if (this.CurrentTable.options.currentPage > 1) {
        this.CurrentTable.options.currentPage--; 
        
      }
    },
    // 监听子组件触发的下一页事件
  handleNextPage() {
      if (this.CurrentTable.options.currentPage < this.totalPages) {
        this.CurrentTable.options.currentPage++; 
      }
    },
  hideNode() {
    if (this.selectedNode) {
      this.hiddenNodes.push(this.selectedNode.name);
      this.closeNodeModal();
      this.updateGraph();
    }
  },
  initNodeFilterData() {
      this.allNodes = this.networkData.nodes.map((node, index) => ({
        key: index,
        label: node.name,
        disabled: false
      }));
      this.visibleNodes = this.allNodes
        .filter(node => !this.hiddenNodes.includes(node.label))
        .map(node => node.key);
    },
    
  filterNodes(query, item) {
      return item.label.toLowerCase().indexOf(query.toLowerCase()) > -1;
    },
    
  applyNodeFilter() {
      const hiddenNodeNames = this.allNodes
        .filter(node => !this.visibleNodes.includes(node.key))
        .map(node => node.label);
      
      this.hiddenNodes = hiddenNodeNames;
      this.updateGraph();
      this.closeNodeFilterModal();
    },
  updateGraph() {
    const visibleNodes = this.networkData.nodes.filter(node => !this.hiddenNodes.includes(node.name));
    const visibleLinks = this.networkData.links.filter(link => 
      !this.hiddenNodes.includes(link.source) && !this.hiddenNodes.includes(link.target)
    );
    this.chart.setOption({
      series: [{
        data: visibleNodes,
        links: visibleLinks,
      }]
    });
  },
  resizeChart() {
    if (this.chart) {
      this.chart.resize();
    }
  },
  getBadgeClass(level) {
        switch(level.toLowerCase()) {
          case 'high': return 'bg-danger';
          case 'medium': return 'bg-warning';
          case 'low': return 'bg-success';
          default: return 'bg-secondary';
        }
  },        
  showImageModal() {
        this.showModalImage = true;
      },
  showTable(){
    this.showTables = true;
  },
  async showReport() {
      this.showModalReport = true;
      await this.$nextTick(); // 等待 DOM 更新
      await this.fetchReports();
      this.initChart(); // 在模态框显示后初始化图表
  },
  closeImageModal() {
        this.showModalImage = false;
  },
  closeReportModal() {
    this.showModalReport = false;
    if (this.chart) {
      this.chart.dispose();
      this.chart = null;
    }
  },
  closeTableModal() {
    this.showTables = false;
  },


      /*async fetchReports() {
  try {
    const response = await axios.get(``);
    this.reportData = response.data;
    this.processNetworkData();
  } catch (error) {
    console.error('Error fetching reports:', error);
    this.reportData = null;
  }
},*/
async fetchReports() {
    // 模拟 API 响应
    const mockResponse = {
      data: {
        "anomalous_actions": {
          "data": [
            {
              "Time": 1522987541537076593,
              "SubjectType": "Netflow",
              "SubjectName": "128.55.12.10:53",
              "Action": "EVENT_RECVFROM",
              "ObjectType": "Subject",
              "ObjectName": "smtpd"
            },
            {
              "Time": 152298754153000000,
              "SubjectType": "Netflow",
              "SubjectName": "128.55.12.10:53",
              "Action": "EVENT_RECVFROM",
              "ObjectType": "Subject",
              "ObjectName": "smtpd"
            },
            {
              "Time": 152298754153700000,
              "SubjectType": "Netflow",
              "SubjectName": "128.55.12.10:53",
              "Action": "EVENT_RECVFROM",
              "ObjectType": "Subject",
              "ObjectName": "smtpd"
            },
            {
              "Time": 1522987541537076000,
              "SubjectType": "Netflow",
              "SubjectName": "128.55.12.10:53",
              "Action": "EVENT_RECVFROM",
              "ObjectType": "Subject",
              "ObjectName": "smtpd"
            },
            {
              "Time": 1522987541537070000,
              "SubjectType": "Netflow",
              "SubjectName": "128.55.12.10:53",
              "Action": "EVENT_RECVFROM",
              "ObjectType": "Subject",
              "ObjectName": "smtpd"
            },
            {
              "Time": 1522988100977066981,
              "SubjectType": "Netflow",
              "SubjectName": "128.55.12.10:53",
              "Action": "EVENT_RECVFROM",
              "ObjectType": "Subject",
              "ObjectName": "imapd"
            },
            {
              "Time": 1522987244081943163,
              "SubjectType": "Subject",
              "SubjectName": "lsof",
              "Action": "EVENT_SENDTO",
              "ObjectType": "Netflow",
              "ObjectName": "128.55.12.10:53"
            }
          ],
          "total": 3
        },
        "dangerous_actions": {
          "data": [],
          "total": 0
        }
      }
    };

    // 使用模拟数据
    this.reportData = mockResponse.data;
    this.AlterTable.data = this.reportData.anomalous_actions.data.map(action => ({
        时间: this.formatTime(action.Time),
        主体类型: action.SubjectType,
        主体名称: action.SubjectName,
        行为: action.Action,
        客体类型: action.ObjectType,
        客体名称: action.ObjectName
      }));

      this.DangerTable.data = this.reportData.dangerous_actions.data.map(action => ({
        时间: this.formatTime(action.Time),
        主体类型: action.SubjectType,
        主体名称: action.SubjectName,
        行为: action.Action,
        客体类型: action.ObjectType,
        客体名称: action.ObjectName
      }));

    this.processNetworkData();
    this.$nextTick(() => {
      this.renderEChartsGraph();
    });
  },


initChart() {
  const container = document.getElementById('echarts-container');
  if (container) {
      this.chart = echarts.init(container);
      this.renderEChartsGraph();
    } 
  else{
    console.error('ECharts container not found');
    }
  },


processNetworkData() {
  if (!this.reportData) return;

  const nodes = new Set();
  const links = [];

  this.reportData.anomalous_actions.data.forEach(action => {
    nodes.add(action.SubjectName);
    nodes.add(action.ObjectName);
    
    const existingLink = links.find(link => 
      link.source === action.SubjectName && link.target === action.ObjectName
    );

    if (existingLink) {
      existingLink.actions.push(action);
    } else {
      links.push({
        source: action.SubjectName,
        target: action.ObjectName,
        actions: [action]
      });
    }
  });

  this.networkData = {
    nodes: Array.from(nodes).map(name => {
      const nodeData = this.reportData.anomalous_actions.data.find(
        action => action.SubjectName === name || action.ObjectName === name
      );
      return {
        name,
        value: name,
        SubjectType: nodeData.SubjectType,
        ObjectType: nodeData.ObjectType
      };
    }),
    links: links.map(link => ({
      source: link.source,
      target: link.target,
      value: link.actions.length,
      actions: link.actions,
      id: `${link.source}-${link.target}`
    }))
  };

  this.$nextTick(() => {
    this.renderEChartsGraph();
  });
},

toggleEdgeStyle() {
    this.isStraightLine = !this.isStraightLine;
    this.renderEChartsGraph(); // 重新渲染图表
  },

renderEChartsGraph() {
  // 找出最大的 action 数量，用于归一化
  const maxActions = Math.max(...this.networkData.links.map(link => link.value));
  const option = {
    tooltip: {
      trigger: 'item',
      formatter: (params) => {
        if (params.dataType === 'edge') {
          return `${params.data.source} -> ${params.data.target}<br/>Actions: ${params.data.value}`;
        }
        return params.name;
      }
    },
    series: [{
      type: 'graph',
      layout: 'force',
      data: this.networkData.nodes.filter(node => !this.hiddenNodes.includes(node.name)),
        links: this.networkData.links.filter(link => 
          !this.hiddenNodes.includes(link.source) && !this.hiddenNodes.includes(link.target)
        ),
      roam: true,
      label: {
        show: true,
        position: 'right',
        formatter: '{b}'
      },
      lineStyle: {
        color: 'source',
        curveness: this.isStraightLine ? 0 : 0.3, // 保持直线
      },
      emphasis: {
        focus: 'adjacency',
        lineStyle: {
          width: 4 , // 鼠标悬停时的线条宽度
          curveness: this.isStraightLine ? 0 : 0.3,
        }
      },
      force: {
        repulsion: 150,
        edgeLength: 250,
        gravity: 0.1
      },
      edgeSymbol: ['circle', 'arrow'],
      edgeSymbolSize: [4, 10],
      draggable: true,
      cursor: 'pointer'
    }]
  };

  this.chart.setOption(option);
  this.addChartClickListeners();
},

calculateLineWidth(value, maxValue) {
  const minWidth = 0.5;  
  const maxWidth = 3;    
  if (maxValue === 0 || value === 0) return minWidth;
  return minWidth + (Math.log(value) / Math.log(maxValue)) * (maxWidth - minWidth);
},
addChartClickListeners() {
  this.chart.on('click', (params) => {
    console.log('Clicked:', params);  
    if (params.dataType === 'node') {
      this.showNodeDetails(params.data);
    } 
    else if (params.dataType === 'edge') {
      this.showEdgeDetails(params.data);
    }
  });
},

showNodeDetails(node) {
    this.selectedNode = {
      name: node.name,
      type: node.SubjectType || node.ObjectType
    };
    this.showNodeModal = true;
  },

showEdgeDetails(edge) {
  console.log('Edge data:', edge);  
  this.selectedEdge = {
    source: edge.source,
    target: edge.target,
    actions: edge.actions || []
  };
  this.showEdgeModal = true;
},


showNodeFilter() {
      this.showNodeFilterModal = true;
      this.initNodeFilterData();
  },
    
closeNodeFilterModal() {
      this.showNodeFilterModal = false;
  },

closeNodeModal() {
    this.showNodeModal = false;
    this.selectedNode = null;
  },

closeEdgeModal() {
    this.showEdgeModal = false;
    this.selectedEdge = null;
  },

prevImage() {
  if (this.currentImageIndex > 0) {
    this.currentImageIndex--;
    }
  },
nextImage() {
  if (this.currentImageIndex < this.images.length - 1) {
      this.currentImageIndex++;
    }
  },
formatTime(timestamp) {
  if (!timestamp) return 'Invalid timestamp';
  const date = new Date(Number(timestamp) / 1000000);
  return isNaN(date.getTime()) ? 'Invalid date' : date.toLocaleString();
},
    }
  };
  </script>
  
  <style scoped>
  .source-child {
    background-color: #f8f9fa;
    border-radius: 5px;
    padding: 20px;
    margin-bottom: 20px;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
  }
  
  .modal-overlay {
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background: rgba(0, 0, 0, 0.8);
    display: flex;
    justify-content: center;
    align-items: center;
    z-index: 1000;
  }
  
  .modal-content {
    background: white;
    padding: 20px;
    border-radius: 5px;
    max-width: 80%;
    max-height: 80%;
    overflow: auto;
    position: relative;
  }
  
  .close-btn {
    position: absolute;
    top: 10px;
    right: 10px;
    font-size: 24px;
    cursor: pointer;
    background: none;
    border: none;
    color: #333;
  }
  
  .modal-navigation {
    display: flex;
    justify-content: space-between;
    margin-top: 20px;
  }
  
  .img-fluid {
    max-width: 100%;
    height: auto;
  }
  .hide-node-btn, .show-all-nodes-btn {
  margin-top: 10px;
  padding: 8px 16px;
  background-color: #f44336;
  color: white;
  border: none;
  border-radius: 4px;
  cursor: pointer;
  font-size: 14px;
}

.hide-node-btn:hover, .show-all-nodes-btn:hover {
  background-color: #d32f2f;
}
.el-transfer {
  text-align: left;
  display: inline-block;
  vertical-align: middle;
  width: 100%;
  height: 400px;
}
.table-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 20px;
}
.table-selector {
  margin-right:20px;
}
  </style>