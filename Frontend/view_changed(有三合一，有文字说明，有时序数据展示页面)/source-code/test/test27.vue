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
          <div class="row ablity-button">
          <button class="btn btn-primary ml-1" @click="showImageModal">查看相关图片</button>
          <button class="btn btn-primary ml-1" @click="showReportModal">查看安全报告</button>
          <button class="btn btn-primary ml-1" @click="showTableModal">查看行为表格</button>
          <button class="btn btn-primary ml-1" @click="showAdviceModal">查看安全建议</button>
        </div>
        </div>
      </div>
  
      <!-- 全屏图片模态框 -->
      <div v-if="showModalImage" class="modal-overlay" @click="closeImageModal">
  <div class="modal-content" @click.stop>
    <button class="close-btn" @click="closeImageModal">&times;</button>
    <div class="statistics-panel-img col-12" v-loading="loadingImg" element-loading-text="加载中...">
      <el-image 
        v-if="currentImage" 
        :src="currentImage" 
        fit="contain"
        :preview-src-list="[currentImage]"
      >
        <div slot="error" class="image-slot">
          <i class="el-icon-picture-outline"></i>
        </div>
      </el-image>
      <span style="display: block; text-align: center;">{{source.文件名}}</span>
    </div>
  </div>
</div>

<div v-if="showModalReport" class="modal-overlay" @click="closeReportModal">
  <div class="modal-content report-modal" @click.stop>
    <button class="close-btn" @click="closeReportModal">&times;</button>
    <h3 class="modal-title">安全报告</h3>
    <div class="report-container row">
      <div class="statistics-panel" v-loading="loadingMsg" element-loading-text="加载中...">
        <el-collapse-transition>
        <div v-show="showMessage">
        <h4>统计信息</h4>
        <p>数据总量: {{ statistics.dataCount }}</p>
        <h5>最危险攻击方:</h5>
        <ul>
          <li v-for="(attacker, index) in statistics.topAttackers" :key="'attacker-'+index">
            {{ attacker.name }}: {{ attacker.count }} 次
          </li>
        </ul>
        <h5>最危险被攻击方:</h5>
        <ul>
          <li v-for="(target, index) in statistics.topTargets" :key="'target-'+index">
            {{ target.name }}: {{ target.count }} 次
          </li>
        </ul>
        <h5>前三种边类型统计：</h5>
      <ul>
       <li v-for="(count, edgeType) in Object.entries(edgeTypeStats).slice(0, 3)" :key="edgeType">
          {{ edgeType }}: {{ count }} 次
        </li>
      </ul>
      </div>
    </el-collapse-transition>
  <el-collapse-transition>
    <div v-show="!showMessage">
      <h4>事件信息</h4>
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
      </div>
  </el-collapse-transition>
  <div class="message-button mb-2">
    <el-button @click="showMessage = !showMessage">切换信息</el-button>
    </div>
  </div>
  <div class="echarts-container" id="echarts-container" v-loading="loadingGra" element-loading-text="加载中...">
    <div v-show="!showSimplifiedGraph" id="full-graph" style="width: 100%; "></div>
    <div v-show="showSimplifiedGraph" id="simplified-graph" ref="b" style="width: 100%; "></div>
  </div>
  
    </div>
    <div class="button-container">
      <button @click="toggleEdgeStyle" class="btn btn-primary mr-3">
        {{ isStraightLine ? '曲线显示' : '直线显示' }}
      </button>
      <button class="btn btn-primary" @click="showNodeFilter">筛选节点</button>
      <button class="btn btn-primary ml-3" @click="toggleGraph">{{ showSimplifiedGraph ? '显示完整溯源图' : '显示关键溯源图' }}</button>
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
      <p><strong>危险等级:</strong> {{ action.危险等级 }}</p>
      <hr v-if="index < selectedEdge.actions.length - 1">
    </div>
  </div>
</div>

    <div v-if="showTables" class="modal-overlay" @click="closeTableModal">
  <div class="modal-content" @click.stop v-if="showTables">
    <button class="close-btn" @click="closeTableModal">&times;</button>
    <div class="col-12">
      <div class="col-12">
    <div class="table-header row justify-content-between">
    <div class="table-selector">
      <label for="tableSelect">选择表格:</label>
      <select id="tableSelect" v-model="selectedTable" class="form-select">
        <option value="AlterTable">可疑行为列表</option>
        <option value="DangerTable">危险行为列表</option>
      </select>
      </div>
      <div class="search-bar mb-2" >
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

<div v-if="showAdvice" class="modal-overlay" @click="closeAdviceModal">
    <div class="modal-content advice-modal" @click.stop>
      <button class="close-btn" @click="closeAdviceModal">&times;</button>
      <h3>安全建议</h3>
      <el-collapse-transition>
        <div v-if="!showDetailedAdvice">
          <p v-for="(advice, index) in securityAdvice" :key="index">{{ advice }}</p>
          <el-button @click="showDetailedSecurityAdvice" type="primary">显示详细安全建议</el-button>
        </div>
      </el-collapse-transition>
      <el-collapse-transition>
        <div v-if="showDetailedAdvice" class="detailed-advice">
          <div class="message">
          <img class="ai-avatar" src="../assets/img/k-logo.png" alt="AI Avatar">
          <div class="advice-content">
            <p>{{ detailedSecurityAdvice.content }}</p>
          </div>
          </div>
          <el-button @click="showDetailedAdvice = false" type="primary">返回简要建议</el-button>
        </div>
      </el-collapse-transition>
    </div>
  </div>
  </div>
  </template>
  
  <script>
  import { mapActions } from 'vuex';
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
        loadingImg:false,
        loadingMsg:false,
        loadingGra:false,

        showSimplifiedGraph: false,
        showAdvice:false,
        showMessage:true,
        showTables:false,
        showModalImage: false,
        showModalReport:false,
        showNodeModal: false,
        showEdgeModal: false,
        showNodeFilterModal:false,
        showDetailedAdvice: false,

        isStraightLine: true,

        selectedNode: null,
        selectedEdge: null,

        reportData: null,
        networkData: null,

        forceRerender: false,

        chart: null,
        hiddenNodes: [],
        allNodes: [],
        visibleNodes: [],
        allFilteredData: [],

        selectedTable: 'DangerTable',
        searchText: '',

        securityAdvice: [],
        edgeTypeStats: {},
        detailedSecurityAdvice: {
          type:'ai',
          content:"",
          typedText: '',
          typeIndex:0,
        },
        statistics: {
          topAttackers: [],
          topTargets: [],
          dataCount: 0
        },
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
      console.log(this.source);
      this.fetchReports();
    },

    beforeDestroy() {
      if (this.chart) {
        this.chart.dispose();
      }
      if (this.simplifiedChart) {
        this.simplifiedChart.dispose();
      }
      window.removeEventListener('resize', this.resizeChart);
    },
    computed: {
  currentImage() {
    const image = this.source.图片内容;
    return image ? `data:image/png;base64,${image}` : '';
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
      toggleGraph() {
        
        this.showSimplifiedGraph = !this.showSimplifiedGraph;
        console.log(this.showSimplifiedGraph);
        
        if (this.showSimplifiedGraph) {
          this.$nextTick(() => {
            this.renderSimplifiedEChartsGraph();
          });
        } 
        else {
          this.$nextTick(() => {
            this.renderEChartsGraph();
          });
        }
        // 更新标题
        this.updateChartTitle();
      },

updateChartTitle() {
  const title = this.showSimplifiedGraph ? '关键溯源图' : '完整溯源图';
  const chart = this.showSimplifiedGraph ? this.simplifiedChart : this.chart;
  chart.setOption({
    title: {
      text: title
    }
  });
},
  async fetchImage() {
  if (!this.source.图片内容) {
    this.loadingImg = true;
    try {
      const response = await axios.get('http://43.138.200.89:8080/kairos/graph/content', {
        params: {
          file_name: this.source.文件名,
        },
        headers: {
          'content-type': 'application/json',
        }
      });
      this.$set(this.source, '图片内容', response.data.file_content);
      // 强制重新计算 currentImage
      this.$nextTick(() => {
        this.forceRerender = !this.forceRerender;
      });
    } catch (error) {
      console.log('溯源图获取出错', error);
    } finally {
      this.loadingImg = false;
    }
  }
},
    async showDetailedSecurityAdvice() {
      this.showDetailedAdvice = true;
      if (!this.detailedSecurityAdvice.content) {
        const securityEvent = {networkData: [this.networkData],
          statistics: [this.statistics]}
        const prompt = `安全事件信息: ${JSON.stringify(securityEvent)}\n用户问题:给出详细安全建议`
        try {
          axios.post('http://43.138.200.89:8080/kairos/completions', 
            { prompt: prompt }
          ).then(response =>{
            const aiResponse = response.data.advice;
            this.detailedSecurityAdvice.content = aiResponse;
            this.typeNextChar(this.detailedSecurityAdvice);
        })
        } catch (error) {
          console.error('Error fetching detailed security advice:', error);
          this.detailedSecurityAdvice.content = '抱歉，获取详细安全建议时出现错误。请稍后再试。';
          this.detailedSecurityAdvice.typedText = this.detailedSecurityAdvice.content;
        }
      } 
      else {
        // 如果已经有内容，重新开始打字动画
        this.detailedSecurityAdvice.typedText = '';
        this.detailedSecurityAdvice.typeIndex = 0;
        this.typeNextChar(this.detailedSecurityAdvice);
      }
    },
    
    typeNextChar(message) {
      if (message.typeIndex < message.content.length) {
        message.typedText += message.content.charAt(message.typeIndex);
        message.typeIndex++;
        setTimeout(() => this.typeNextChar(message), 50);
      }
    }, 
calculateStatistics() {
  const attackers = {};
  const targets = {};
  this.networkData.links.forEach(link => {
    attackers[link.source] = (attackers[link.source] || 0) + 1;
    targets[link.target] = (targets[link.target] || 0) + 1;
  });

  const sortNodes = (obj) => Object.entries(obj)
    .sort((a, b) => b[1] - a[1])
    .map(([name, count]) => ({ name, count }));

  const topCount = this.networkData.links.length < 1000 ? 3 : 6;
  this.edgeTypeStats = {};
  this.networkData.links.forEach(link => {
    link.actions.forEach(action => {
      this.edgeTypeStats[action.Action] = (this.edgeTypeStats[action.Action] || 0) + 1;
    });
  });

  // 对边类型进行排序
  this.edgeTypeStats = Object.entries(this.edgeTypeStats)
    .sort((a, b) => b[1] - a[1])
    .reduce((obj, [key, value]) => {
      obj[key] = value;
      return obj;
    }, {});
  this.statistics = {
    topAttackers: sortNodes(attackers).slice(0, topCount),
    topTargets: sortNodes(targets).slice(0, topCount),
    dataCount: this.networkData.links.length
  };
},
generateSecurityAdvice() {
    this.securityAdvice = [];
    // 对攻击方的建议
    console.log(this.statistics);
    if (this.statistics.topAttackers.length > 0) {
      const topAttacker = this.statistics.topAttackers[0].name;
      this.securityAdvice.push(`${topAttacker} 节点作为最频繁的攻击方较为危险，建议加强其防护和监控。`);
    }
    // 对被攻击方的建议
    if (this.statistics.topTargets.length > 0) {
      const topTarget = this.statistics.topTargets[0].name;
      this.securityAdvice.push(`${topTarget} 节点作为最常被攻击的目标较为脆弱，建议对其进行全面检查与必要的隔离措施。`);
    }
    const edgeTypes = [
    'EVENT_RECVFROM', 'EVENT_SENDTO', 'EVENT_EXECUTE', 
    'EVENT_WRITE', 'EVENT_OPEN', 'EVENT_CLOSE'
  ];

  edgeTypes.forEach(edgeType => {
    const count = this.edgeTypeStats[edgeType] || 0;
    let advice = '';
    switch (edgeType) {
      case 'EVENT_RECVFROM':
        if (count > 0) {
          advice = `检测到 ${count} 次接收事件，建议审查入站流量，确保数据来源可信，并检查是否存在未经授权的数据接收。`;
        }
        break;
      case 'EVENT_SENDTO':
        if (count > 0) {
          advice = `发现 ${count} 次发送事件，建议监控出站连接，防止可能的数据外泄，并确保所有外发数据都经过适当的加密和授权。`;
        }
        break;
      case 'EVENT_EXECUTE':
        if (count > 0) {
          advice = `存在 ${count} 次执行事件，建议审查所有被执行的程序，确保它们都是经过授权和安全的。考虑实施应用程序白名单策略。`;
        }
        break;
      case 'EVENT_WRITE':
        if (count > 0) {
          advice = `检测到 ${count} 次写入事件，建议监控文件系统变更，防止恶意软件写入或重要文件被篡改。考虑实施文件完整性监控。`;
        }
        break;
      case 'EVENT_READ':
        if (count > 0) {
          `检测到 ${count} 次读取事件，建议审查文件访问权限，确保敏感数据不被未授权访问。考虑实施数据加密、访问日志记录，并监控异常的读取模式，如大量或频繁读取敏感文件。同时，评估是否存在数据泄露的风险，并确保所有读取操作都有正当理由。`;
        }
      case 'EVENT_OPEN':
        if (count > 0) {
          advice = `有 ${count} 次文件或资源打开事件，建议审查访问控制策略，确保只有授权用户和进程能够访问敏感资源。`;
        }
        break;
      case 'EVENT_CLOSE':
        if (count > 0) {
          advice = `记录到 ${count} 次关闭事件，虽然这通常是正常行为，但建议检查是否有异常的资源使用模式，例如频繁的打开后立即关闭。`;
        }
        break;
    }
    if (advice) {
      this.securityAdvice.push(advice);
    }
  });
    // 根据数据量添加额外建议
    if (this.statistics.dataCount > 1000) {
      this.securityAdvice.push("鉴于大量的交互行为，建议加强整体网络安全策略，并考虑实施更严格的访问控制措施。");
    } 
    else {
      this.securityAdvice.push("虽然交互行为数量不多，仍建议定期检查和更新安全策略，以防潜在威胁。");
    }
  },


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
  if (this.simplifiedChart) {
    this.simplifiedChart.resize();
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
        this.fetchImage();
      },
  showTableModal(){
    this.showTables = true;
  },
  async showReportModal() {
      this.showModalReport = true;
      await this.$nextTick(); // 等待 DOM 更新
      await this.fetchReports();
      this.initChart(); // 在模态框显示后初始化图表
  },
  showAdviceModal(){
      this.generateSecurityAdvice();
      this.showAdvice = true;
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
  closeAdviceModal(){
    this.showAdvice = false;
  },

async fetchReports() {
    this.loadingMsg = true;
    this.loadingGra = true;
    try {
    let startTime = new Date(this.source.开始时间.replace(/-/g, '/'));
    let endTime = new Date(this.source.结束时间.replace(/-/g, '/'));
    if (startTime.getTime() === endTime.getTime()) {
      endTime.setSeconds(endTime.getSeconds() + 1);
    }
    const formatTime = date => {
      const pad = num => (num < 10 ? '0' : '') + num;
      return `${date.getFullYear()}-${pad(date.getMonth() + 1)}-${pad(date.getDate())} ${pad(date.getHours())}:${pad(date.getMinutes())}:${pad(date.getSeconds())}`;
    };

    const formattedStartTime = formatTime(startTime);
    const formattedEndTime = formatTime(endTime);
    this.source.结束时间 = formattedEndTime;
    console.log(formattedStartTime,formattedEndTime);
    const response_kairos = await axios.get('http://43.138.200.89:8080/kairos/graph-actions', {
        params: {
          graph_index:this.source.文件名.slice(0,-4),
        },
        headers: {
          'content-type':'application/json',
        }
      });
    this.reportData = response_kairos.data;
    this.AlterTable.data = this.reportData.anomalous_actions.data.map(action => ({
      时间: this.formatTime(action.Time),
      主体类型: action.SubjectType,
      主体名称: action.SubjectName,
      行为: action.Action,
      客体类型: action.ObjectType,
      客体名称: action.ObjectName
    }));
    if(this.reportData.dangerous_actions.data == null){
      this.reportData.dangerous_actions.data = [];
    }
    console.log(this.reportData);
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
  } catch (error) {
    console.error('Error fetching reports:', error);
    // 可以在这里添加错误处理逻辑
  } finally {
    this.loadingMsg = false;
    this.loadingGra = false;
  }
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
    if (!this.reportData) {
    console.error('报告数据不完整');
    return;
  }

  const nodes = new Set();
  const links = new Map();

  const processAction = (action, dangerLevel) => {
    nodes.add(action.SubjectName);
    nodes.add(action.ObjectName);
    
    const linkKey = `${action.SubjectName}-${action.ObjectName}`;
    if (!links.has(linkKey)) {
      links.set(linkKey, {
        source: action.SubjectName,
        target: action.ObjectName,
        actions: [],
        hasDangerousAction: false
      });
    }
    const link = links.get(linkKey);
    link.actions.push({...action, 危险等级: dangerLevel});
    if (dangerLevel === '危险') {
      link.hasDangerousAction = true;
    }
  };
    this.reportData.anomalous_actions.data.forEach(action => processAction(action, '可疑'));
    this.reportData.dangerous_actions.data.forEach(action => processAction(action, '危险'));
  this.networkData = {
    nodes: Array.from(nodes).map(name => {
      const nodeData = [...this.reportData.anomalous_actions.data, ...this.reportData.dangerous_actions.data]
        .find(action => action.SubjectName === name || action.ObjectName === name);
      return {
        name,
        value: name,
        SubjectType: nodeData.SubjectType,
        ObjectType: nodeData.ObjectType
      };
    }),
    links: Array.from(links.values()).map(link => ({
      ...link,
      value: link.actions.length,
      id: `${link.source}-${link.target}`
    }))
  };

  console.log('Processed network data:', this.networkData);
  this.calculateStatistics();
},

toggleEdgeStyle() {
    this.isStraightLine = !this.isStraightLine;
    this.renderEChartsGraph(); // 重新渲染图表
  },

  renderEChartsGraph() {
  const container = document.getElementById('full-graph');
  if (!this.chart) {
    this.chart = echarts.init(container);
  }
  const maxActions = Math.max(...this.networkData.links.map(link => link.value));
  const option = {
    title: {
      text: '完整溯源图',  // 或 '关键溯源图'
      left: 'center',
      top: 10,
      textStyle: {
        fontSize: 18,
        fontWeight: 'bold'
      }
    },
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
      layout: 'none',
      data: this.networkData.nodes.filter(node => !this.hiddenNodes.includes(node.name)),
      links: this.networkData.links.filter(link =>
        !this.hiddenNodes.includes(link.source) && !this.hiddenNodes.includes(link.target)
      ).map(link => ({
        ...link,
        lineStyle: {
          color: this.getLinkColor(link)
        }
      })),
      roam: true,
      label: {
        show: false,
        position: 'right',
        formatter: '{b}',
        fontSize: 12
      },
      lineStyle: {
        curveness: this.isStraightLine ? 0 : 0.3,
      },
      emphasis: {
        focus: 'adjacency',
        lineStyle: {
          width: 4,
        },
        label: {
          show: true
        }
      },
      edgeSymbol: ['circle', 'arrow'],
      edgeSymbolSize: [4, 8],
      draggable: true,
      cursor: 'pointer',
      symbolSize: (value, params) => {
        const connectedLinks = this.networkData.links.filter(link =>
          link.source === params.name || link.target === params.name
        );
        return Math.min(10 + connectedLinks.length * 2, 30);
      },
      itemStyle: {
        color: '#4169E1',
        borderColor: '#000',
        borderWidth: 1
      }
    }]
  };

  this.setNodePositions(option.series[0].data);

  this.chart.setOption(option);
  this.addChartClickListeners();
  this.loadingGra = false;
},

// 修改后的方法：获取连接颜色
getLinkColor(link) {
  return link.hasDangerousAction ? "#FF0000" : "#0000FF";
},

// 修改后的方法：设置节点位置
setNodePositions(nodes) {
  const totalNodes = nodes.length;
  const radius = Math.max(totalNodes * 50, 1000000); 
  
  nodes.forEach((node, index) => {
    const angle = (index / totalNodes) * 2 * Math.PI;
    node.x = radius * Math.cos(angle);
    node.y = radius * Math.sin(angle);
  });
},
renderSimplifiedEChartsGraph() {
  const container = this.$refs.b;
  if (!this.simplifiedChart) {
    this.simplifiedChart = echarts.init(container);
  }
  let simplifiedNodes = JSON.parse(JSON.stringify(this.networkData.nodes));
  let simplifiedLinks = JSON.parse(JSON.stringify(this.networkData.links));
  const irrelevantFileTypes = ['mount', 'PIPE', 'tmp', '.conf', '.so', '.swp', 'default'];
  const importantOperations = ['.sh', 'mv', 'cp', 'rm'];
  simplifiedNodes = simplifiedNodes.filter(node => {
    if (importantOperations.some(op => node.name.includes(op))) {
      return true;
    }
    return !irrelevantFileTypes.some(type => node.name.includes(type));
  });

  const retainedNodeNames = simplifiedNodes.map(node => node.name);
  simplifiedLinks = simplifiedLinks.filter(link =>
    retainedNodeNames.includes(link.source) && retainedNodeNames.includes(link.target)
  );
  const connectedNodeNames = new Set([
    ...simplifiedLinks.map(link => link.source),
    ...simplifiedLinks.map(link => link.target)
  ]);
  simplifiedNodes = simplifiedNodes.filter(node => connectedNodeNames.has(node.name));

  const option = {
    title: {
      text: '关键溯源图', 
      left: 'center',
      top: 10,
      textStyle: {
        fontSize: 18,
        fontWeight: 'bold'
      }
    },
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
      layout: 'none',
      data: simplifiedNodes,
      links: simplifiedLinks.map(link => ({
        ...link,
        lineStyle: {
          color: this.getLinkColor(link)
        }
      })),
      roam: true,
      label: {
        show: true,
        position: 'right',
        formatter: '{b}',
        fontSize: 10
      },
      lineStyle: {
        curveness: this.isStraightLine ? 0 : 0.3,
      },
      emphasis: {
        focus: 'adjacency',
        lineStyle: {
          width: 4,
        },
        label: {
          show: true
        }
      },
      edgeSymbol: ['circle', 'arrow'],
      edgeSymbolSize: [4, 8],
      draggable: true,
      cursor: 'pointer',
      symbolSize: (value, params) => {
        const connectedLinks = simplifiedLinks.filter(link =>
          link.source === params.name || link.target === params.name
        );
        return Math.min(10 + connectedLinks.length * 2, 30);
      },
      itemStyle: {
        color: '#4169E1',
        borderColor: '#000',
        borderWidth: 1
      }
    }]
  };

  // 设置节点位置
  this.setNodePositions(option.series[0].data);

  // 设置图表选项
  this.simplifiedChart.setOption(option);

  // 添加点击事件监听器
  this.addSimplifiedChartClickListeners();
},

addSimplifiedChartClickListeners() {
  this.simplifiedChart.on('click', (params) => { 
    if (params.dataType === 'node') {
      this.showNodeDetails(params.data);
    } 
    else if (params.dataType === 'edge') {
      this.showEdgeDetails(params.data);
    }
  });
},
calculateLineWidth(value, maxValue) {
  const minWidth = 1;
  const maxWidth = 5;
  if (maxValue === 0 || value === 0) return minWidth;
  return minWidth + (Math.log(value) / Math.log(maxValue)) * (maxWidth - minWidth);
},

addChartClickListeners() {
  this.chart.on('click', (params) => { 
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
  const hasDangerousAction = edge.actions.some(action => 
    this.reportData.dangerous_actions.data.some(dangerousAction => 
      action.Time === dangerousAction.Time &&
      action.SubjectName === dangerousAction.SubjectName &&
      action.ObjectName === dangerousAction.ObjectName
    )
  );
  
  this.selectedEdge = {
    source: edge.source,
    target: edge.target,
    actions: edge.actions || [],
    dangerLevel: hasDangerousAction ? '危险' : '可疑'
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
    background: rgb(234, 234, 234);
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
.modal-content.report-modal {
  width: 90%;
  height: 90%;
  max-width: 1200px;
  max-height: 800px;
  display: flex;
  flex-direction: column;
  padding: 20px;
}

.modal-title {
  margin-top: 0;
  margin-bottom: 15px;
}

.report-container {
  display: flex;
  flex: 1;
  min-height: 0;
}

.statistics-panel {
  width: 25%;
  padding: 10px;
  background-color: #f0f0f0;
  border-right: 1px solid #ddd;
  overflow-y: auto;
  box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
}

.statistics-panel h4, .statistics-panel h5 {
  margin-top: 0;
  margin-bottom: 10px;
}

.statistics-panel ul {
  padding-left: 20px;
  margin-bottom: 10px;
}

#echarts-container {
  flex: 1;
  min-width: 0;
}

.button-container {
  margin-top: 15px;
  text-align: center;
}
.echarts-container{
  box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
}
.message-button{
  position: absolute;
  bottom: 0;
  left: 50%;
  transform: translateX(-50%);
  width: 100%;
  text-align: center;
}
.statistics-panel {
  position: relative;
  padding-bottom: 60px; /* 留出按钮的位置 */
}
.ablity-button{
  text-align: center;
}
.search-bar {
  display: flex;
  align-items: center;
  margin-right: 20px;
}

.el-button {
  margin-top: 15px;
}

.detailed-advice {
  display: flex;
  flex-direction: column;
  align-items: flex-start;
}

.ai-avatar {
  width: 40px;
  height: 40px;
  border-radius: 50%;
  margin-bottom: 10px;
}

.advice-content {
    max-width: 100%;
    padding: 4px 4px; /* 增加内边距 */
    border-radius: 18px; /* 增加圆角 */
    word-wrap: break-word;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.15);
}
.message {
    margin-bottom: 15px;
    display: flex;
    align-items: center; /* 使头像和消息内容垂直对齐 */
  }

  </style>