<template>
<div class="main-container">
  <div class="d-flex justify-content-end mb-3">
      <el-button @click="toggleView" type="primary">
        {{ showTable ? '显示规则' : '显示表格' }}
      </el-button>
    </div>
  <div class="rule-engine" v-if="!showTable">
    <div class="d-flex justify-content-end">
      <el-button @click="showUploadDialog" type="primary" class="mr-3 btn btn-primary mt-3">导入规则</el-button>
      <button class="btn btn-primary mt-3 mr-3" @click="addEmptyRule">添加新规则</button>
    </div>
    <div class="col-12 mb-2">
      <rule-child v-for="(rule,index) in rules" 
      :key="index" 
      :rule="rule"  
      :rule-id="rule.id"
      :index = "index"
      @delete="deleteRule"
      @updateRules="updateRules"/>
    </div>
    <div v-if="true"><button @click="clearLocalStorage">清空本地存储</button>
      <button @click="showLocalStorage">显示本地存储</button>
    </div>
    <el-dialog :visible.sync="dialogVisible" title="选择文件并上传" width="50%" :modal-append-to-body="false">
            <el-upload
              class="upload-demo"
              ref="upload"
              :on-preview="handlePreview"
              :on-remove="handleRemove"
              :http-request="handleUpload"
              :on-change="handleChange"
              :file-list="fileList"
              :accept='".yaml"'
              :auto-upload="false"
              multiple
            >
              <!-- 插槽内容 -->
              <el-button slot="trigger" size="small" type="primary" class="mt-0">选取文件</el-button>
            </el-upload > 
            <span slot="footer" class="dialog-footer row justify-content-end">
              <el-button style="margin-left: 10px;"  size="small" type="success" class="mb-0" @click="submitUpload">
                确 认
              </el-button>
              <el-button @click="dialogVisible = false">取 消</el-button>
            </span>
          </el-dialog>
          </div>
          <div v-else class="table">
      <div class="col-12 d-flex justify-content-end">
        <div class="table-header col-3">
          <div class="search-bar mb-2">
            <input v-model="searchText" type="text" placeholder="搜索..." class="form-control" />
            <p-button class="search-btn ti-search ml-2"></p-button>
          </div>
        </div>
      </div>
      <div class="col-12">
        <card :title="tableData.title" :subTitle="tableData.subTitle">
          <div slot="raw-content" class="table">
            <paper-table 
              :data="filteredData" 
              :columns="tableData.columns">
            </paper-table>
          </div>
        </card>
        
        <div class="page_button" style="display: flex; justify-content: space-between; align-items: center;">
          <p-button type="info" round @click.native="handlePrevPage" style="margin-right: 10px;">上一页</p-button>
          <span>第 {{ currentPage }} 页 / 共 {{ totalPages }} 页</span>
          <p-button type="info" round @click.native="handleNextPage" style="margin-left: 10px;">下一页</p-button>
        </div>
      </div>
    </div>
  </div>
  </template>
  <script>
  import { StatsCard, ChartCard,PaperTable } from "@/components/index";
  import Chartist from "chartist";
  import RuleChild from "@/components/RuleChild.vue"
  import axios from 'axios';
  export default {
    components:{
      RuleChild,
      StatsCard,
      PaperTable,
      ChartCard,
      Chartist,
    },
    data() {
  return {
    allFilteredData: [],
    fileList: [],
    dialogVisible: false,
    searchText: '',
    showTable: false,  // 控制是否显示表格
    rules: JSON.parse(localStorage.getItem('rules')) || [
      {ruleName:"sql注入", ruleMessage:"1号", multilineText: '', enable:true, warnLevel:"high", deleted: false}, 
      {ruleName:"55555", ruleMessage:"2号", multilineText: '', enable:true, warnLevel:"low", deleted: false},
      {}
    ],
    tableData: {
      title: "预警列表",
      subTitle: "",
      columns: ["ID", "创建时间", "更新时间", "目标名称","目标类型"],
      data: [
        { ID:0,创建时间: "2024-07-17 10:00", 更新时间: "IP", 目标名称: "192.168.1.1", 目标类型: "高" },
        { ID:0,创建时间: "2024-07-17 10:00", 更新时间: "IP", 目标名称: "192.168.1.1", 目标类型: "高" },
        { ID:0,创建时间: "2024-07-17 10:00", 更新时间: "IP", 目标名称: "192.168.1.1", 目标类型: "高" },
        { ID:0,创建时间: "2024-07-17 10:00", 更新时间: "IP", 目标名称: "192.168.1.1", 目标类型: "高" },
        { ID:0,创建时间: "2024-07-17 10:00", 更新时间: "IP", 目标名称: "192.168.1.1", 目标类型: "高" },
        { ID:0,创建时间: "2024-07-17 10:00", 更新时间: "IP", 目标名称: "192.168.1.1", 目标类型: "高" },
        { ID:0,创建时间: "2024-07-17 10:00", 更新时间: "IP", 目标名称: "192.168.1.1", 目标类型: "高" },
        { ID:0,创建时间: "2024-07-17 10:00", 更新时间: "IP", 目标名称: "192.168.1.1", 目标类型: "高" },
        { ID:0,创建时间: "2024-07-17 10:00", 更新时间: "IP", 目标名称: "192.168.1.1", 目标类型: "高" },
        { ID:0,创建时间: "2024-07-17 10:00", 更新时间: "IP", 目标名称: "192.168.1.1", 目标类型: "高" },
        { ID:0,创建时间: "2024-07-17 10:00", 更新时间: "IP", 目标名称: "192.168.1.1", 目标类型: "高" },
        { ID:0,创建时间: "2024-07-17 10:00", 更新时间: "IP", 目标名称: "192.168.1.1", 目标类型: "高" },
        { ID:11,创建时间: "2024-07-17 10:00", 更新时间: "IP", 目标名称: "192.168.1.1", 目标类型: "高" },
      ],
      options: {
        pageSize: 10,
        currentPage: 1,
      }
    }
  }
},
  computed:{
    currentPage() {
    return this.tableData.options.currentPage;
  },
  pageSize() {
    return this.tableData.options.pageSize;
  },
  totalItems() {
    return this.allFilteredData.length;
  },
  totalPages() {
    return Math.ceil(this.totalItems / this.pageSize) || 1;
  },
  filteredData() {
    const startIndex = (this.currentPage - 1) * this.pageSize;
    const endIndex = startIndex + this.pageSize;
    return this.allFilteredData.slice(startIndex, endIndex);
  },
  },
  mounted() {
    this.fetchBlackList();
    if (!localStorage.getItem('rules')) {
      this.saveRules(); // 如果本地存储中没有数据，则保存默认数据
    }
  },
  watch: {
  searchText(newValue) {
    this.updateAllFilteredData();
  },
},
  methods: {
    updateAllFilteredData() {
    console.log("过滤数据",this.allFilteredData);
    const searchText = this.searchText.trim().toLowerCase();
    if (!searchText) {
      this.allFilteredData = this.tableData.data;
      console.log("过滤数据",this.allFilteredData);
    } else {
      this.allFilteredData = this.tableData.data.filter(item => {
        return Object.values(item).some(value => {
          return String(value).toLowerCase().includes(searchText);
        });
      });
    }
  },
    toggleView() {
    this.showTable = !this.showTable;
    if(this.showTable){

    }
  },

    handlePreview(file) {
  const reader = new FileReader();
  reader.onload = (e) => {
    try {
      // 解析 JSON
      const content = JSON.parse(e.target.result);
      const formattedContent = JSON.stringify(content, null, 2);
      
      // 使用 pre 标签来保留格式
      const htmlContent = `<pre style="white-space: pre-wrap; word-wrap: break-word;">${formattedContent}</pre>`;
      
      this.$alert(htmlContent, '文件内容预览', {
        dangerouslyUseHTMLString: true,
        customClass: 'json-preview-dialog',
        closeOnClickModal: true,
        closeOnPressEscape: true,
      });
    } catch (error) {
      this.$message.error('无法解析 JSON 文件: ' + error.message);
    }
  };
  reader.readAsText(file.raw);
},


handleChange(file, fileList) {
  this.fileList = fileList.filter(file => file.name.endsWith('.yaml'));
  if (this.fileList.length !== fileList.length) {
    this.$message.warning('只支持 .yaml 文件，非 yaml 文件已被过滤');
  }
},

handleRemove(file, fileList) {
  this.fileList = fileList;
  this.$message({
    message: `文件 ${file.name} 已被移除`,
    type: 'info'
  });
},


    handleUpload() {
      // 阻止默认的上传行为
      return Promise.resolve();
    },
    submitUpload() {
  if (this.fileList.length === 0) {
    this.$message.warning('请先选择文件');
    return;
  }
  
  const fileReadPromises = this.fileList.map(file => this.readFileAsJSON(file.raw));
  
  Promise.all(fileReadPromises)
    .then(results => {
      results.forEach(result => {
        if (result.success) {
          this.importRule(result.data);
        } else {
          this.$message.error(`文件 ${result.fileName} 导入失败：${result.error}`);
        }
      });
      
      this.dialogVisible = false;
      this.fileList = [];
    })
    .catch(error => {
      console.error('处理文件时发生错误:', error);
      this.$message.error('导入过程中发生错误，请检查文件格式');
    });
},

readFileAsJSON(file) {
  return new Promise((resolve) => {
    const reader = new FileReader();
    reader.onload = (e) => {
      try {
        const importedRule = JSON.parse(e.target.result);
        resolve({ success: true, data: importedRule, fileName: file.name });
      } catch (error) {
        resolve({ success: false, error: '文件不是有效的yaml格式', fileName: file.name });
      }
    };
    reader.readAsText(file);
  });
},


importRule(rule) {
  if (rule.ruleName && rule.warnLevel && rule.ruleMessage && rule.multilineText) {
    const newRule = {
      ruleName: rule.ruleName,
      warnLevel: rule.warnLevel,
      ruleMessage: rule.ruleMessage,
      multilineText: typeof rule.multilineText === 'object' ? JSON.stringify(rule.multilineText, null, 2) : rule.multilineText,
      enable: rule.enable !== undefined ? rule.enable : true,
      showTextArea: false
    };

    // 检查规则名称是否唯一
    if (this.rules.some(existingRule => existingRule.ruleName === newRule.ruleName)) {
      this.$message.warning(`规则 "${newRule.ruleName}" 已存在，将被跳过`);
      return;
    }

    // 将新规则添加到规则数组中
    this.rules.push(newRule);
    // 保存到本地存储
    this.saveRules();

    this.$message.success(`规则 "${newRule.ruleName}" 导入成功！`);
  } else {
    throw new Error('JSON文件格式不正确或缺少必要字段');
  }
},


    showUploadDialog() {
        this.dialogVisible = true;
      }, 


    clearLocalStorage() {
    localStorage.clear();
    this.rules = []; 
  },

  showLocalStorage(){
    console.log(JSON.parse(localStorage.getItem('rules')));
    console.log(this.rules);
  },


  async deleteRule(ruleId) {
    const ruleIndex = this.rules.findIndex(rule => rule.id === ruleId);
  if (ruleIndex === -1) {
    console.error('未找到对应的规则');
    return;
  }
  const deletedRule = this.rules[ruleIndex];
  deletedRule.deleted = true;
  const dataToSend = `${deletedRule.multilineText}
{
  "ruleMessgae":[
    {
      "ruleName": ${deletedRule.ruleName}
      "warnLevel":${deletedRule.warnLevel}
      "ruleMessage":${deletedRule.ruleMessage}
      "deleted":true
    }
  ]
}`;
  this.rules.splice(ruleIndex, 1);
  try{
  const response = await axios.post('http://43.138.200.89:8080/blacklist/set-blacklist', dataToSend);
  console.log('规则成功发送到后端:', response.data);
}catch(error){
  console.log(error);
}
  console.log(this.rules);
  this.removeRuleFromLocalStorage(ruleId);
  this.saveRules();
  this.loadRules();
},

// 新增方法：从本地存储中移除特定规则
removeRuleFromLocalStorage(ruleId) {
  localStorage.removeItem(`ruleName_${ruleId}`);
  localStorage.removeItem(`warnLevel_${ruleId}`);
  localStorage.removeItem(`ruleMessage_${ruleId}`);
  localStorage.removeItem(`multilineText_${ruleId}`);
  localStorage.removeItem(`enable_${ruleId}`);
},


    
saveRules() {
      localStorage.setItem('rules', JSON.stringify(this.rules));
},


    addEmptyRule() {
      const newRule = {
         id: Date.now(), 
         ruleName: "",
         ruleMessage: "",
         multilineText: '',
         enable: false,
         warnLevel: "",
         deleted: false };
      this.rules.push(newRule);
      this.saveRules();
      this.loadRules();
  },


    loadRules() {
      const storedRules = JSON.parse(localStorage.getItem('rules')) || [];
      this.rules = Array.isArray(storedRules) ? storedRules : [];
    },


    updateRules() {
      // 更新 rules 数据，重新从本地存储中加载
      this.loadRules();
    },
    handlePrevPage() {
      if (this.tableData.options.currentPage > 1) {
        this.tableData.options.currentPage--; 
      }
    },
    handleNextPage() {
      if (this.tableData.options.currentPage < this.totalPages) {
        this.tableData.options.currentPage++; 
      }
    },
    async fetchBlackList() {
    try {
      const response = await axios.get('http://43.138.200.89:8080/blacklist/get-blacklist');
      const blackList = response.data.slice(0, 1537); // 假设你想限制返回的数据数量为1537
      const newTableData = [];
      
      for (const item of blackList) {
        newTableData.push({
          ID: item.ID,
          创建时间: this.formatTime(item.CreatedAt),
          更新时间: this.formatTime(item.UpdatedAt),
          目标名称: item.TargetName,
          目标类型: item.TargetType
        });
      }
      
      this.tableData.data = newTableData;
      this.updateAllFilteredData(); // 更新过滤后的数据
    } catch (error) {
      console.error('获取黑名单告警失败:', error);
    }
  },
  }
}
  </script>
  <style>
.rule-engine{
  box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
}
.search-bar {
  display: flex;
  align-items: center;
  margin-right: auto;
}
.json-preview-dialog .el-message-box__message {
  max-height: 70vh;
  overflow: auto;
}

.json-preview-dialog .el-message-box__message pre {
  margin: 0;
  font-family: monospace;
  font-size: 14px;
}
</style>