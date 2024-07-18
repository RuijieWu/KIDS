<template>
  <div class="main-container">
    <div class="top-row">
      <div class="top-row-left">
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
<div class="row align-items-center ml-3" v-if="selectedUploadType === 'file'">
  <el-upload
class="upload-demo row ml-3"
ref="upload"
action="https://jsonplaceholder.typicode.com/posts/"
:on-preview="handlePreview"
:on-remove="handleRemove"
:file-list="fileList"
:show-file-list="true" 
:accept="['.csv', '.xls', '.xlsx']"
:auto-upload="false">
<el-button slot="trigger" size="small" type="primary" class="mt-0">选取文件</el-button>
<el-button style="margin-left: 10px;" size="small" type="success" class="mt-0" @click="submitUpload">上传到服务器</el-button>
</el-upload>
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
      </div>
      <div class="top-row-right">
      <!-- statsCard 部分 -->
      <div class="row">
        <div class="col-md-6 col-xl-3" v-for="(statsGroup, index) in statsCardGroups" :key="index">
          <stats-card>
            <div class="icon-big text-center" :class="`icon-${statsGroup[0].type}`" slot="header">
              <i :class="statsGroup[0].icon"></i>
            </div>
            <div class="numbers" slot="content">
              <p>{{ statsGroup[0].title }}</p>
              <p>{{ statsGroup[0].subTitle }}</p>
              {{ statsGroup[0].value }}
            </div>
          </stats-card>
          <stats-card v-if="statsGroup[1]" class="mt-4">
            <div class="icon-big text-center" :class="`icon-${statsGroup[1].type}`" slot="header">
              <i :class="statsGroup[1].icon"></i>
            </div>
            <div class="numbers" slot="content">
              <p>{{ statsGroup[1].title }}</p>
              <p>{{ statsGroup[1].subTitle }}</p>
              {{ statsGroup[1].value }}
            </div>
          </stats-card>
        </div>
      </div>
    </div>
  </div>
    <div class="bottom-row">
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
  </div>
</template>

<script>
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
    StatsCard,
    PaperTable,
    ChartCard,
    Chartist,

  },
  data() {
    return {
      showTextArea: false,
      selectedUploadType: 'file',
      selectedFile: null,
      manualInput: '',
      allFilteredData: [],
      selectedTable: 'AttackerTable',
      searchText: '',
      uploadMethod:[{
        value:'file',
        label:'文件上传'
      },{
        value:'manual',
        label:'手动输入'
      }
      ],
      statsCards: [
      {
        type: "warning",
        icon: "ti-server",
        title: "攻击方",
        subTitle:"(可疑/危险)",
        value: "5/1",
        footerIcon: "ti-reload",
      },
      {
        type: "success",
        icon: "ti-wallet",
        title: "被攻击方",
        subTitle:"(可疑/危险)",
        value: "5/5",
        footerIcon: "ti-calendar",
      },
      {
        type: "danger",
        icon: "ti-pulse",
        title: "异常行为",
        subTitle:"(可疑/危险)",
        value: "2897/0",
        footerIcon: "ti-timer",
      },
      {
        type: "info",
        icon: "ti-twitter-alt",
        title: "任务",
        subTitle:"()",
        value: "10",
        footerIcon: "ti-reload",
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
      }
    };
  },
  computed: {
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
      const searchText = this.searchText.trim().toLowerCase();
      return this.CurrentTable.data.filter(item => {
        return Object.values(item).some(value => {
          return String(value).toLowerCase().includes(searchText);
        });
      });
    },
    statsCardGroups() {
    const groups = [];
    for (let i = 0; i < this.statsCards.length; i += 2) {
      groups.push([this.statsCards[i], this.statsCards[i + 1]]);
    }
    return groups;
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
    submitUpload() {
      this.$refs.upload.submit();
    },
    changeShow(){
      this.showTextArea = !this.showTextArea;
      return this.showTextArea;
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
      this.AttackerTable.data = this.AttackerTable.data.concat(newData);
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
      const sourceSaveIntervalDays = parseInt(localStorage.getItem('sourceSaveInterval')) || 0;
      const now = new Date();
      const startTime = new Date(now.getTime() - sourceSaveIntervalDays * 24 * 60 * 60 * 1000);
      const formattedStartTime = this.formatDate_date(startTime);
      const formattedEndTime = this.formatDate_date(now);
      const response_kairos = await axios.get('http://43.138.200.89:8080/kairos/actions', {
        params: {
          start_time: '2018-04-01 00:00:00',
          end_time: '2018-04-12 00:00:00',
        },
        headers: {
          "content-type": "application/json",
        }
      });
      const response = await axios.get('http://43.138.200.89:8080/alarm/message/list', {
        params: {
          page: this.AlterTable.options.currentPage.toString(),
          size: '12',
          alarmTypes: '11,10,8,14,15,16,13,9,12',
          attackStartTime: formattedStartTime,
          attackEndTime: formattedEndTime,
          securitys: '1,2,3',
        },
        headers: {
          "content-type": "application/json",
        }
      });
      const securityMap = { '1': '低危', '2': '中危', '3': '高危' };
      const alarmTypeMap = { '11': '异常登录', '10': '暴力破解', '8': '可疑操作', '14': '动态蜜罐', '15': 'webshell', '16': '系统后门', '13': '反弹shell', '9': 'web命令执行', '12': '本地提权' };
      const alarms = response.data.body.content;
      this.AlterTable.data = alarms.map(item => ({
        时间: item.hostname,
        主体类型: item.publicIp,
        主体名称: item.publicIp,
        行为: alarmTypeMap[item.alarmType],
        客体类型: item.viewPointVO.attackLocation,
        客体名称: item.viewPointVO.attackIp,
      }));
      const newAlterData = [];
      const newDangerData = [];
      for (const subject of response_kairos.data.anomalous_actions.data) {
        newAlterData.push({
          时间: formatTime(subject.Time),
          主体类型: subject.SubjectType,
          主体名称: subject.SubjectName,
          行为: subject.Action,
          客体类型: subject.ObjectType,
          客体名称: subject.ObjectName,
        });
      }
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
      this.AlterTable.data = this.AttackerTable.data.concat(newAlterData);
      this.DangerTable.data = this.DangerTable.data.concat(newDangerData);
    },
    handleFileChange(event) {
      this.selectedFile = event.target.files[0];
    },
    async uploadFile() {
      if (this.selectedFile) {
        const formData = new FormData();
        formData.append('file', this.selectedFile);

        try {
          const response = await axios.post('/api/upload', formData, {
            headers: {
              'Content-Type': 'multipart/form-data'
            }
          });
          alert('文件上传成功：' + response.data.message);
          this.selectedFile = null;
          this.$refs.fileInput.value = '';
        } catch (error) {
          console.error('上传错误：', error);
          alert('上传失败，请重试。');
        }
      } else {
        alert('请选择要上传的文件。');
      }
    },
    async sendManualInput() {
      if (this.manualInput.trim()) {
        try {
          const response = await axios.post('/api/manual-input', { data: this.manualInput });
          alert('数据发送成功：' + response.data.message);
          this.manualInput = '';
        } catch (error) {
          console.error('发送错误：', error);
          alert('发送失败，请重试。');
        }
      } else {
        alert('请输入要发送的数据。');
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
overflow-y: auto;
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
