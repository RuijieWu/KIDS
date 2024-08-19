<template>
<div class="main-container">
  <div class="d-flex justify-content-end mb-3">
      <el-button @click="toggleView" type="primary">
        {{ showTable ? '显示规则' : '显示表格' }}
      </el-button>
    </div>
  <div class="rule-engine" v-if="!showTable">
    <div class="d-flex justify-content-between">
      <div class="mt-3 ml-2">
      <el-select v-model="ruleTypeFilter" placeholder="筛选规则类型" style="width: 150px;">
          <el-option label="全部" value="all"></el-option>
          <el-option label="黑名单" value="black"></el-option>
          <el-option label="白名单" value="white"></el-option>
        </el-select>
      </div>
        <div>
      <el-button @click="showUploadDialog" type="primary" class="mr-3 btn btn-primary mt-3">导入规则</el-button>
      <el-button class="btn btn-primary mt-3 mr-3" type="primary" @click="addEmptyRule">添加新规则</el-button>
    </div>
  </div>
    <div class="col-12 mb-2">
      <rule-child v-for="(rule,index) in filteredRules" 
      :key="index" 
      :rule="rule"  
      :ruleId="rule.id"
      :index = "index"
      @delete="deleteRule"
      @updateRules="updateRules"/>
    </div>
    <!--
    <div v-if="true"><button @click="clearLocalStorage">清空本地存储</button>
      <button @click="showLocalStorage">显示本地存储</button>
    </div>
    -->
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
              :key="`${tableData.title}-${dataUpdateTrigger}`"
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
    dataUpdateTrigger: 0,
    ruleTypeFilter: 'all',
    allFilteredData: [],
    fileList: [],
    dialogVisible: false,
    searchText: '',
    showTable: false,  // 控制是否显示表格
    rules: JSON.parse(localStorage.getItem('rules')) || [
      {ruleName:"sql注入", ruleMessage:"1号", multilineText: '', enable:true, warnLevel:"high", deleted: false,ruleType:"black"}, 
      {ruleName:"55555", ruleMessage:"2号", multilineText: '', enable:true, warnLevel:"low", deleted: false,ruleType:"white"},
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
  filteredRules() {
      if (this.ruleTypeFilter === 'all') {
        return this.rules;
      } else {
        return this.rules.filter(rule => rule.ruleType === this.ruleTypeFilter);
      }
    }
  },
  mounted() {
    this.fetchBlackList();
    if (!localStorage.getItem('rules')) {
      this.saveRules(); // 如果本地存储中没有数据，则保存默认数据
    }
    else{
      this.loadRules();
    }
    this.setupAudit();
  },
  watch: {
  searchText(newValue) {
    this.updateAllFilteredData();
  },
},
  methods: {
    async setupAudit(){
      try{
        const response = await axios.post('http://43.138.200.89:8080/data/setup-audit','{"paths":["/home/ubuntu/softbei/KIDS/Backend", "/home/ubuntu/softbei/KIDS/Frontend"]}')
      }catch(error){
        console.log("set up audit error:",error);
      }
    },

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
    this.dataUpdateTrigger += 1;
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
      const content = JSON.parse(e.target.result);
      const formattedContent = JSON.stringify(content, null, 2);
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
      showTextArea: false,
      ruleType :rule.ruleType,
      id:rule.id,
    };
    if (this.rules.some(existingRule => existingRule.ruleName === newRule.ruleName)) {
      this.$message.warning(`规则 "${newRule.ruleName}" 已存在，将被跳过`);
      return;
    }
    this.rules.push(newRule);
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
  console.log("存储",JSON.parse(localStorage.getItem('rules')));
  console.log("数组",this.rules);
},


  async deleteRule(ruleId) {
    const ruleIndex = this.rules.findIndex(rule => rule.id === ruleId);
  if (ruleIndex === -1) {
    console.error('未找到对应的规则');
    return;
  }
  const deletedRule = this.rules[ruleIndex];
  deletedRule.deleted = true;
  const dataToSend = `"ruleText":{
  ${deletedRule.multilineText}
}
{
  "ruleMessgae":[
    {
      "ruleName": ${deletedRule.ruleName}
      "warnLevel":${deletedRule.warnLevel}
      "ruleMessage":${deletedRule.ruleMessage}
      "deleted":true,
      "ruleType":${deletedRule.ruleType}
    }
  ]
}`;
  this.rules.splice(ruleIndex, 1);
  try{
  if(this.deleteRule.ruleType == "black"){
    const response = await axios.post('http://43.138.200.89:8080/blacklist/set-blacklist', dataToSend);
    console.log('规则成功发送到后端:', response.data);
  }
  else if(this.deleteRule.ruleType == "white"){
    const response = await axios.post('http://43.138.200.89:8080/blacklist/set-whitelist', dataToSend);
    console.log('规则成功发送到后端:', response.data);
  }
}catch(error){
  console.log(error);
}
  console.log(this.rules);
  this.removeRuleFromLocalStorage(ruleId);
  this.saveRules();
  this.loadRules();
},

removeRuleFromLocalStorage(ruleId) {
  localStorage.removeItem(`ruleName_${ruleId}`);
  localStorage.removeItem(`warnLevel_${ruleId}`);
  localStorage.removeItem(`ruleMessage_${ruleId}`);
  localStorage.removeItem(`multilineText_${ruleId}`);
  localStorage.removeItem(`enable_${ruleId}`);
  localStorage.removeItem(`ruleType_${ruleId}`);
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
         deleted: false,
         ruleType: "black", 
        };
      console.log(newRule.ruleType);
      this.rules.push(newRule);
      this.saveRules();
      this.loadRules();
  },


    loadRules() {
      const storedRules = JSON.parse(localStorage.getItem('rules')) || [];
      this.rules = Array.isArray(storedRules) ? storedRules : [];
    },


    updateRules() {
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
      const blackList = response.data;
      const newTableData = [];
      
      for (const item of blackList) {
        console.log(item);
        newTableData.push({
          ID: item.ID,
          创建时间: item.CreatedAt.slice(0,19),
          更新时间: item.UpdatedAt.slice(0,19),
          目标名称: item.TargetName,
          目标类型: item.TargetType
        });
      }
      
      this.tableData.data = newTableData;
      this.updateAllFilteredData(); 
      this.dataUpdateTrigger += 1;
      this.searchText = '1';
      this.searchText = "";
    } catch (error) {
      console.error('获取黑名单告警失败:', error);
    }
  },
  },
  beforeDestroy(){
    this.saveRules();
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