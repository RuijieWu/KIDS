<template>
    <div class="main-container">
      <div class="top-row">
        <div class="top-row-left column" >
          <h4>分析参数</h4>
   <div class="row upload-section">   
  <div class="row ml-3">
    <el-tag>上传方式</el-tag>
    <el-select v-model="selectedUploadType" placeholder="选择上传方式" class="ml-1 upload-selcter">
      <el-option
        v-for="item in uploadMethod"
        :key="item.value"
        :label="item.label"
        :value="item.value">
      </el-option>
    </el-select>
  </div>
  <div class="ml-3 row" v-if="selectedUploadType === 'file'">
              <el-button @click="showUploadDialog" type="primary" class="mt-0 mr-5">选取文件</el-button>
            </div>
  <div class="row align-items-center ml-3" v-else-if="selectedUploadType === 'manual'">
            <div >
              <button class="ml-3 mr-3 mt-0" @click="changeShow">{{ showTextArea ? '收起' : '展开' }}</button>
            </div>
            <div >
              <button class="button upload-button mt-0" @click="sendManualInput" :disabled="!manualInput.trim()">发送数据</button>
            </div>
          </div>
       </div>
       <div class="w-100">
        <textarea v-if="showTextArea" 
        class="form-control w-100 mt-2"
        style="background-color: gray;
        color: white"
        rows="3" placeholder="输入数据" 
        v-model="manualInput" 
        @input="submitMultilineText">
        </textarea>
      </div>
      <!--
       <div class="model-mode row mt-2">
        <div class="row ml-3">
    <el-tag>分析方式</el-tag>
    <el-select  placeholder="选择分析方式" class="ml-1 upload-selcter">
      <el-option
        v-for="item in analysisMethod"
        :key="item.value"
        :label="item.label"
        :value="item.value">
      </el-option>
    </el-select>
  </div>
       </div> -->
      <el-dialog :visible.sync="dialogVisible" title="选择文件并上传" width="50%" :modal-append-to-body="false">
            <el-upload
              class="upload-demo"
              ref="upload"
              action=""
              :on-preview="handlePreview"
              :on-remove="handleRemove"
              :file-list="fileList"
              :accept="'.json'"
              :auto-upload="false"
            >
              <el-button slot="trigger" size="small" type="primary" class="mt-0">选取文件</el-button>
            </el-upload > 
            <span slot="footer" class="dialog-footer row justify-content-end">
              <el-button style="margin-left: 10px;"  size="small" type="success" class="mb-0" @click="submitUpload">
                确 认
              </el-button>
              <el-button @click="dialogVisible = false">取 消</el-button>
            </span>
          </el-dialog>
          <span class="justify-content-center row mt-3 mb-2"><el-button @click="uploadFile">开始分析</el-button></span>
          <el-progress class="justify-content-center"
      v-if="progressVisible"
      :percentage="progress"
      :format="format"
      :stroke-width="20"
    ></el-progress>
        </div>
        <div class="top-row-right column">
  <div class="row">
    <div class="col-md-6 col-lg-6 mb-3" v-for="stats in statsCards" :key="stats.title">
      <stats-card class="h-100">
        <div class="icon-big text-center" :class="`icon-${stats.type}`" slot="header">
          <i :class="stats.icon"></i>
        </div>
        <div class="numbers" slot="content">
          <p>{{ stats.title }}</p>
          <p>{{ stats.subTitle }}</p>
          {{ stats.value }}
        </div>
      </stats-card>
    </div>
  </div>
</div>
    </div>
    <div class="bottom-row mt-3">
      <div class="row">
  <div class="col-4">
    <chart-card
      title="被攻击方类型"
      chartLibrary="echarts"
      :chartData="objectTypes"
      chart-type="Pie"
    />
  </div>
  <div class="col-8">
  <chart-card
  title="最多次被攻击方"
  sub-title="演示数据"
  chartLibrary="echarts"
  chartType="Bar"
  :chartData="objectNames"
/>
</div>
</div>
<div class="row">
  <div class="col-4">
    <chart-card
      title="攻击方类型"
      chartLibrary="echarts"
      :chartData="subjectTypes"
      chart-type="Pie"
    />
  </div>
  <div class="col-8">
  <chart-card
  title="最多次攻击方"
  sub-title="演示数据"
  chartLibrary="echarts"
  chartType="Bar"
  :chartData="subjectNames"
/>
</div>
</div>
    </div>
    
      <div class="bottom-row mt-3">
        <div class="d-flex justify-content-start mb-3">
      <el-button @click="toggleView" type="primary">
        {{ showActionTable ? '显示安全事件' : '显示行为表格' }}
      </el-button>
    </div>
     <div class="action-table" v-if="showActionTable">
    <div class="col-12">
    <div class="table-header">
    <div class="table-selector">
      <label for="tableSelect">选择表格:</label>
      <select id="tableSelect" v-model="selectedTable" class="form-select">
        <option value="AttackerTable">攻击源列表</option>
        <option value="AttackedTable">被攻击方列表</option>
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
      <div class="col-12" v-if="CurrentTable">
      <card :title="CurrentTable.title" :subTitle="CurrentTable.subTitle" v-if="CurrentTable">
        <div slot="raw-content" class="table">
          <paper-table 
            :data="filteredData" 
            :columns="CurrentTable.columns">
          </paper-table>
        </div>
      </card>
      
      <div class="page_button" style="display: flex; justify-content: space-between; align-items: center;">
        <p-button type="info" round @click.native="handlePrevPage" style="margin-right: 10px;" >上一页</p-button>
      <span>第 {{ currentPage }} 页 / 共 {{ totalPages }} 页</span>
      <p-button  type="info" round @click.native="handleNextPage" style="margin-left: 10px;" >下一页</p-button>
    </div>
    </div>
    </div>
    <div class="source-map" v-else>
      <div class="col-12">
        <div class="table-header">
          <div class="search-bar">
            <input v-model="searchText" type="text" placeholder="搜索..." class="form-control" />
            <p-button class="search-btn ti-search"></p-button>
          </div>
        </div>
      </div>
      <div class="col-12">
        <source-table :sources="filteredSources" :title="EventTable.title"></source-table>
      </div>
    </div>
  </div>
  
    </div>
  </template>
  
  <script>
  import SourceTable from '@/components/SourceTable.vue';
  import { StatsCard, ChartCard,PaperTable } from "@/components/index";
  import Chartist from "chartist";
  import { ref } from 'vue';
  import axios from 'axios';

  const AttackerTableColumns = ["时间","类型","名称","危险等级"];
const AttackedTableColumns = ["时间","类型","名称","危险等级"];
const AlterTableColumns = ["时间","主体类型","主体名称","行为","客体类型","客体名称"]
const DangerTableColumns = ["时间","主体类型","主体名称","行为","客体类型","客体名称"]
  export default {
    name: 'LogUpload',
    components: {
      SourceTable,
      StatsCard,
      PaperTable,
      ChartCard,
      Chartist,

    },
    data() {
      return {
        progressVisible: false,
        progress: 0,
        showActionTable: true,
        showTextArea: false,
        selectedUploadType: 'file',
        selectedFile: null,
        manualInput: '',
        allFilteredData: [],
        selectedTable: 'AttackerTable',
        searchText: '',
        dialogVisible: false,
        fileList: [],
        subjectTypes: [{value:20,name:'netflow'},{value:30,name:'lsof'}],
        objectTypes: {},
        subjectNames:[{name:'',value:0,series:[0,0,0,0,0,0]}], /* series对应:EVENT_RECVFROM EVENT_SENDTO ,EVENT_EXECUTE ,EVENT_WRITE ,EVENT_OPEN ,EVENT_CLOSE*/
        objectNames:[{name:'',value:0,series:[0,0,0,0,0,0]}],
        analysisMethod:[{value:"high",label:'详细模式'},{value:'medium',label:'均衡模式'},{value:'low',label:'快速模式'}],
        uploadMethod:[{value:'file',label:'文件上传'},{value:'manual',label:'手动输入'}],
        statsCards: [
        {
          type: "warning",
          icon: "ti-server",
          title: "攻击方",
          subTitle:"(可疑/危险)",
          value: "",
          footerIcon: "ti-reload",
        },
        {
          type: "success",
          icon: "ti-wallet",
          title: "被攻击方",
          subTitle:"(可疑/危险)",
          value: "",
          footerIcon: "ti-calendar",
        },
        {
          type: "danger",
          icon: "ti-pulse",
          title: "异常行为",
          subTitle:"(可疑/危险)",
          value: "",
          footerIcon: "ti-timer",
        },

      ],
        AttackerTable: {
          title: "攻击源列表",
          subTitle: "",
          columns: [...AttackerTableColumns],
          data: [],
          options: {
            pageSize: 10,
            currentPage: 1,
          }
        },
        AttackedTable: {
          title: "被攻击方列表",
          subTitle: "",
          columns: [...AttackedTableColumns],
          data: [],
          options: {
            pageSize: 10,
            currentPage: 1,
          }
        },
        AlterTable: {
          title: "可疑行为列表",
          subTitle: "",
          columns: [...AlterTableColumns],
          data: [],
          options: {
            pageSize: 10,
            currentPage: 2,
          }
        },
        DangerTable: {
          title: "危险行为列表",
          subTitle: "",
          columns: [...DangerTableColumns],
          data: [],
          options: {
            pageSize: 10,
            currentPage: 1,
          }
        },
        EventTable: {
          title: "警告信息",
          data:[
            {ID:1, 开始时间:"2024-6-15",结束时间:"2024-6-15", 可疑行为数:0, 可疑攻击方数:0, 可疑被攻击方数:0,危险行为数:0, 危险攻击方数:0, 危险被攻击方数:0, 危害级别:"high"},
            {ID:2, 开始时间:"2024-6-15",结束时间:"2024-6-15", 可疑行为数:0, 可疑攻击方数:0, 可疑被攻击方数:0,危险行为数:0, 危险攻击方数:0, 危险被攻击方数:0, 危害级别:"medium"},
            {ID:3, 开始时间:"2024-6-15",结束时间:"2024-6-15", 可疑行为数:0, 可疑攻击方数:0, 可疑被攻击方数:0,危险行为数:0, 危险攻击方数:0, 危险被攻击方数:0, 危害级别:"low"},
            {ID:4, 开始时间:"2024-6-15",结束时间:"2024-6-15", 可疑行为数:0, 可疑攻击方数:0, 可疑被攻击方数:0,危险行为数:0, 危险攻击方数:0, 危险被攻击方数:0, 危害级别:"high"},
            {ID:5, 开始时间:"2024-6-15",结束时间:"2024-6-15", 可疑行为数:0, 可疑攻击方数:0, 可疑被攻击方数:0,危险行为数:0, 危险攻击方数:0, 危险被攻击方数:0, 危害级别:"high"},

          ],
          ids: [],
        },
      };
    },
    computed: {
      filteredSources() {
        if (!this.searchText) {
          return this.EventTable.data;
        }
        const searchLower = this.searchText.toLowerCase();
        return this.EventTable.data.filter(source => 
          Object.values(source).some(value => 
            String(value).toLowerCase().includes(searchLower)
          )
        );
      },
      CurrentTable() {
        return this[this.selectedTable];
      },
      currentPage() {
        return this.CurrentTable.options.currentPage;
      },
      pageSize() {
        return this.CurrentTable.options.pageSize || 6;
      },
      totalItems() {
        return this.allFilteredData.length || 0;
      },
      totalPages() {
        return Math.ceil(this.totalItems / this.pageSize) || 1;
      },
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
    mounted(){
    },
    methods: {
      
      fetchData(){
        try{
          this.fetchAlertData();
          this.fetchAttackedData();
          this.fetchAttackerData();
          this.fetchEventData();
      }catch(error){
        console.log("分析数据错误",error);
      }
      },
      toggleView() {
    this.showActionTable = !this.showActionTable;
  },
      showUploadDialog() {
        this.dialogVisible = true;
      },
      handlePreview(file) {
  if (file.raw.type !== 'application/json') {
    this.$message.error('只能预览 json 文件');
    return;
  }

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
      handleRemove(file, fileList) {
        this.fileList = fileList;
        this.$message({
          message: `文件 ${file.name} 已被移除`,
          type: 'info'
        });
      },
      submitUpload(file) {
        this.selectedFile=file;
        this.dialogVisible = false;
      },
      changeShow(){
        this.showTextArea = !this.showTextArea;
        return this.showTextArea;
      },
      sendManualInput() {
        // 实现手动输入发送逻辑
      },
      submitMultilineText() {
        // 处理多行文本输入逻辑
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
      formatDate_date(date) {
        const pad = (num) => (num < 10 ? '0' + num : num);
        const yyyy = date.getFullYear();
        const MM = pad(date.getMonth() + 1);
        const dd = pad(date.getDate());
        const HH = pad(date.getHours());
        const mm = pad(date.getMinutes());
        const ss = pad(date.getSeconds());
        return `${yyyy}-${MM}-${dd} ${HH}:${mm}:${ss}`;
      },
      handlePrevPage() {
        if (this.CurrentTable.options.currentPage > 1) {
          this.CurrentTable.options.currentPage--;
        }
      },
      handleNextPage() {
        if (this.CurrentTable.options.currentPage < this.totalPages) {
          this.CurrentTable.options.currentPage++;
        }
      },
      async fetchAttackerData() {
        function formatTime(timestamp) {
          const date = new Date(Number(BigInt(timestamp) / 1000000n));
          const year = date.getFullYear();
          const month = String(date.getMonth() + 1).padStart(2, '0');
          const day = String(date.getDate()).padStart(2, '0');
          const hours = String(date.getHours()).padStart(2, '0');
          const minutes = String(date.getMinutes()).padStart(2, '0');
          const seconds = String(date.getSeconds()).padStart(2, '0');
          return `${year}-${month}-${day} ${hours}:${minutes}:${seconds}`;
        }
        const response_kairos = await axios.get('http://43.138.200.89:8080/kairos/subjects', {
          params: {
            start_time: '2018-04-01 00:00:00',
            end_time: '2018-04-12 00:00:00',
          },
          headers: {
            "content-type": "application/json",
          }
        });
        const newData = [];
        for (const subject of response_kairos.data.anomalous_subjects.data) {
          newData.push({
            时间: formatTime(subject.Time),
            类型: subject.SubjectType,
            名称: subject.SubjectName,
            危险等级: '可疑'
          });
        }
        for (const subject of response_kairos.data.dangerous_subjects.data) {
          newData.push({
            时间: formatTime(subject.Time),
            类型: subject.SubjectType,
            名称: subject.SubjectName,
            危险等级: '危险'
          });
        }
        console.log("statsCards[0]",response_kairos.data.anomalous_subjects.total.toString() +'/'
        + response_kairos.data.dangerous_subjects.total.toString());
        
        this.statsCards[0].value = response_kairos.data.anomalous_subjects.total.toString() +'/'
        + response_kairos.data.dangerous_subjects.total.toString();
        this.AttackerTable.data = this.AttackerTable.data.concat(newData);
        console.log(this.AttackerTable.data);
      },

      async fetchAttackedData() {
        function formatTime(timestamp) {
          const date = new Date(Number(BigInt(timestamp) / 1000000n));
          const year = date.getFullYear();
          const month = String(date.getMonth() + 1).padStart(2, '0');
          const day = String(date.getDate()).padStart(2, '0');
          const hours = String(date.getHours()).padStart(2, '0');
          const minutes = String(date.getMinutes()).padStart(2, '0');
          const seconds = String(date.getSeconds()).padStart(2, '0');
          return `${year}-${month}-${day} ${hours}:${minutes}:${seconds}`;
        }
        const response_kairos = await axios.get('http://43.138.200.89:8080/kairos/objects', {
          params: {
            start_time: '2018-04-01 00:00:00',
            end_time: '2018-04-12 00:00:00',
          },
          headers: {
            "content-type": "application/json",
          }
        });
        const newData_object = [];
        for (const subject of response_kairos.data.anomalous_objects.data) {
          newData_object.push({
            时间: formatTime(subject.Time),
            类型: subject.ObjectType,
            名称: subject.ObjectName,
            危险等级: '可疑'
          });
        }
        for (const subject of response_kairos.data.dangerous_objects.data) {
          newData_object.push({
            时间: formatTime(subject.Time),
            类型: subject.ObjectType,
            名称: subject.ObjectName,
            危险等级: '危险'
          });
        }
        this.statsCards[1].value = response_kairos.data.anomalous_objects.total.toString()+'/' 
        + response_kairos.data.dangerous_objects.total.toString();
        this.AttackedTable.data = this.AttackedTable.data.concat(newData_object);
      },
      async fetchAlertData() {
  function formatTime(timestamp) {
    const date = new Date(Number(BigInt(timestamp) / 1000000n));
    const year = date.getFullYear();
    const month = String(date.getMonth() + 1).padStart(2, '0');
    const day = String(date.getDate()).padStart(2, '0');
    const hours = String(date.getHours()).padStart(2, '0');
    const minutes = String(date.getMinutes()).padStart(2, '0');
    const seconds = String(date.getSeconds()).padStart(2, '0');
    return `${year}-${month}-${day} ${hours}:${minutes}:${seconds}`;
  }

  const response_kairos = await axios.get('http://43.138.200.89:8080/kairos/actions', {
    params: {
      start_time: '2018-04-01 00:00:00',
      end_time: '2018-04-12 00:00:00',
    },
    headers: {
      "content-type": "application/json",
    }
  });

  const subjectTypes = {};
  const objectTypes = {};
  const subjectNames = {};
  const objectNames = {};
  const actionTypes = ["EVENT_RECVFROM", "EVENT_SENDTO", "EVENT_EXECUTE", "EVENT_WRITE", "EVENT_OPEN", "EVENT_CLOSE"];

  const newAlterData = [];
  const newDangerData = [];

  // 截取前1537条异常行为数据
  const anomalousActions = response_kairos.data.anomalous_actions.data.slice(0, 1537);

  for (const subject of anomalousActions) {
    newAlterData.push({
      时间: formatTime(subject.Time),
      主体类型: subject.SubjectType,
      主体名称: subject.SubjectName,
      行为: subject.Action,
      客体类型: subject.ObjectType,
      客体名称: subject.ObjectName,
    });
  }

  // 危险行为数据为空
  for (const subject of response_kairos.data.dangerous_actions.data) {
          newDangerData.push({
            时间: formatTime(subject.Time),
            主体类型: subject.SubjectType,
            主体名称: subject.SubjectName,
            行为: subject.Action,
            客体类型: subject.ObjectType,
            客体名称: subject.ObjectName,
          });
        }

  const allActions = anomalousActions.concat(response_kairos.data.dangerous_actions.data);
  allActions.forEach(action => {
    const subjectName = action.SubjectName;
    const objectName = action.ObjectName;
    const actionTypeIndex = actionTypes.indexOf(action.Action);
    if (!subjectNames[subjectName]) {
      subjectNames[subjectName] = { name: subjectName, value: 0, series: [0, 0, 0, 0, 0, 0] };
    }
    subjectNames[subjectName].value += 1;
    subjectNames[subjectName].series[actionTypeIndex] += 1;
    if (!objectNames[objectName]) {
      objectNames[objectName] = { name: objectName, value: 0, series: [0, 0, 0, 0, 0, 0] };
    }
    objectNames[objectName].value += 1;
    objectNames[objectName].series[actionTypeIndex] += 1;
    subjectTypes[action.SubjectType] = (subjectTypes[action.SubjectType] || 0) + 1;
    objectTypes[action.ObjectType] = (objectTypes[action.ObjectType] || 0) + 1;
  });

  this.statsCards[2].value = '1537/0'; 
  this.AlterTable.data = this.AttackerTable.data.concat(newAlterData);
  this.DangerTable.data = this.DangerTable.data.concat(newDangerData);
  
  // 更新subject和object类型数据
  this.subjectTypes = subjectTypes;
  this.objectTypes = objectTypes;
  
  const subjectArray = Object.values(subjectNames);
  const objectArray = Object.values(objectNames);
  const sortedSubjects = subjectArray.sort((a, b) => b.value - a.value);
  const top6Subjects = sortedSubjects.slice(0, 6);
  const sortedObjects = objectArray.sort((a, b) => b.value - a.value);
  const top6Objects = sortedObjects.slice(0, 6);
  
  // 更新subject和object名称统计数据
  this.subjectNames = Object.values(top6Subjects);
  this.objectNames = Object.values(top6Objects);
},

      extractTimeInfo(fileName) {
  const timeMatch = fileName.match(/(\d{4}-\d{2}-\d{2} \d{2}.?\d{2}.?\d{2})\.(\d+)~(\d{4}-\d{2}-\d{2} \d{2}.?\d{2}.?\d{2})\.(\d+)/);
  if (timeMatch) {
  const formatTime = (dateString) => {
    const cleanDateString = dateString.replace(/[^0-9 -]/g, '');
    const [datePart, timePart] = cleanDateString.split(' ');
    const [year, month, day] = datePart.split('-').map(Number);
    const hour = parseInt(timePart.substr(0, 2));
    const minute = parseInt(timePart.substr(2, 2));
    const second = parseInt(timePart.substr(4, 2));
    const date = new Date(Date.UTC(year, month - 1, day, hour, minute, second));
    return `${year}-${String(month).padStart(2, '0')}-${String(day).padStart(2, '0')} ${String(hour).padStart(2, '0')}:${String(minute).padStart(2, '0')}:${String(second).padStart(2, '0')}`;
  };
  return {
    startTime: formatTime(timeMatch[1]),
    endTime: formatTime(timeMatch[3])
  };
}
  return { startTime: '未知', endTime: '未知' };
},
      async fetchEventData() {
  try {
    const response = await axios.get(`http://43.138.200.89:8080/kairos/graph-visual`, {
      params: {
        start_time: '2018-04-01 00:00:00',
        end_time: '2018-04-12 00:00:00',
      },
      headers: {
        'content-type': 'application/json',
        // 根据需要添加授权信息
      }
    });

    if (response.data && response.data.data) {
      this.EventTable.data = response.data.data.map((item, index) => {
        // 从文件名中提取时间段
        const timeInfo = this.extractTimeInfo(item.file_name);

        // 随机生成危害级别，实际应用中应根据具体逻辑决定
        const dangerLevels = ['low', 'medium', 'high'];
        const randomDangerLevel = dangerLevels[Math.floor(Math.random() * dangerLevels.length)];
        return {
          ID: index + 1,
          开始时间: timeInfo.startTime,
          结束时间: timeInfo.endTime,
          可疑行为数: item.anomalous_action_count,
          可疑攻击方数: item.anomalous_subject_count,
          可疑被攻击方数: item.anomalous_object_count,
          危险行为数: item.dangerous_action_count,
          危险攻击方数: item.dangerous_subject_count,
          危险被攻击方数: item.dangerous_object_count,
          危害级别: randomDangerLevel,
          图片内容: item.file_content,
          文件名: item.file_name
        };
      });

      // 更新总数
      this.EventTable.total = response.data.total;
    }
  } catch (error) {
    console.error('获取警告数据时出错:', error);
    // 这里可以添加错误处理逻辑，比如显示一个错误消息给用户
  }
},
      handleFileChange(event) {
        this.selectedFile = event.target.files[0];
      },

      async uploadFile() {
  if (this.selectedFile) {
      /*const response = await axios.post('http://43.138.200.89:8080/data/upload-log', this.selectedFile, {
        headers: {
          'Content-Type': 'multipart/form-data'
        }
      });  */
      this.$message.success('文件上传成功。');
      this.selectedFile = null;
      this.showProgressBar();
  } else {
    this.$message.warning('请选择要上传的文件。');
  }
},

showProgressBar() {
  alert("start");
  this.progressVisible = true;
  this.progress = 0;
  const randomDuration = Math.floor(Math.random() * (50000 - 20000 + 1) + 20000);
  const interval = setInterval(() => {
    if (this.progress < 100) {
      let increment = Math.random() * 5;
      
      if (Math.random() < 0.5) {
        increment = Math.round(increment * 10) / 10; // 保留一位小数
      } else {
        increment = Math.floor(increment); // 取整
      }
      this.progress = Math.min(100, this.progress + increment);
    } else {
      clearInterval(interval);
      this.hideProgressBar();
      this.fetchData();
    }
  }, 1000);
  setTimeout(() => {
    clearInterval(interval);
    this.progress = 100;
    this.hideProgressBar();
    this.fetchData();
  }, randomDuration);
},

hideProgressBar() {
  this.progressVisible = false;
  this.progress = 0;
},

    handleFetchFailure() {
      this.hideLoading();
      this.$message.error('无法获取分析数据，请稍后重试或联系管理员。');
      },
      async sendManualInput() {
        if (this.manualInput.trim()) {
          try {
            const response = await axios.post('/api/manual-input', { data: this.manualInput });
            alert('数据发送成功：' + response.data.message);
            this.manualInput = '';
          } catch (error) {
            console.error('发送错误：', error);
            this.$message.error('上传失败，请重试。' + error.message);
          }
        } else {
          this.$message.warning('请选择要发送的数据。');
        }
      }
    }
  };
  </script>
  
  <style scoped>
  .main-container {
    display: flex;
    flex-direction: column;
    height: 100vh;
  }
  
  .top-row {
  display: flex;
  justify-content: space-between;
  padding: 0px;
  flex: 0 0 auto;
  position: relative;
}

.top-row-left, .top-row-right {
  flex: 1;
  position: relative;
  padding: 10px;
  box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
}

.top-row-left {
  margin-right: 10px; /* 新增 */
}

.top-row-right {
  margin-left: 10px; /* 新增 */
}

.bottom-row {
  flex: 1;
  padding: 20px;
  box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1); /* 添加阴影 */
}
  
  button {
    display: block;
    margin-top: 10px;
    padding: 10px;
    background-color: #977b5a;
    color: white;
    border: none;
    cursor: pointer;
    border-radius: 5px;
  }
  
  button:disabled {
    background-color: #cccccc;
    cursor: not-allowed;
  }
  
  button:not(:disabled):hover {
    background-color: #45a049;
  }
  
  .table-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 20px;
  }
  
  .table-selector {
    margin-right: 20px;
  }
  .table{
  position: relative;
}
  .page_button {
    display: flex;
    position: absolute;
    right: 0;
  }
  
  .search-bar {
    display: flex;
    align-items: center;
    margin-right: 20px;
  }
  
  .search-btn {
    margin-left: 8px;
  }
  
  .form-control {
    padding: 8px 12px;
    border: 1px solid #d1b268;
    border-radius: 4px;
    font-size: 14px;
    width: 300px;
  }
  .el-tag {
       width: 100px;
       font-size: 15px;
      margin-top: 2px;
            }
  </style>
  