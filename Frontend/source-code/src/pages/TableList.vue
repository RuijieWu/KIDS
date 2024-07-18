<template>
  <div class="row">
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
      
      <div class="page_button">
        <p-button type="info" round @click.native="handlePrevPage" >上一页</p-button>
      <span>第 {{ currentPage }} 页 / 共 {{ totalPages }} 页</span>
      <p-button type="info" round @click.native="handleNextPage" >下一页</p-button>
    </div>
    </div>
  </div>
</template>
<script>
import { PaperTable } from "@/components";
import axios from "axios";
const AttackerTableColumns = ["时间","类型","名称","危险等级"];
const AttackedTableColumns = ["时间","类型","名称","危险等级"];
const AlterTableColumns = ["时间","主体类型","主体名称","行为","客体类型","客体名称"]
const DangerTableColumns = ["时间","主体类型","主体名称","行为","客体类型","客体名称"]
export default {
  components: {
    PaperTable,
  },
  data() {
    return {
      allFilteredData: [],
      selectedTable: 'AlterTable',
      searchText: '',
      AttackerTable: {
        title: "攻击源列表",
        subTitle: "",
        columns: [...AttackerTableColumns],
        data: [],
        options:{
          pageSize: 10, 
          currentPage: 1, 
        }
      },
      AttackedTable:{
        title: "被攻击方列表",
        subTitle: "",
        columns: [...AttackedTableColumns],
        data:[],
        options:{
          pageSize: 10, 
          currentPage: 1, 
        }
      },
      AlterTable: {
        title: "可疑行为列表",
        subTitle: "",
        columns: [...AlterTableColumns],
        data: [],
        options:{
          pageSize: 10, 
          currentPage: 2, 
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
    };
  },
  mounted() {
    const sourceUpdateIntervalMinutes = parseInt(localStorage.getItem('sourceUpdateInterval')) || 2;
    const sourceUpdateIntervalMilliseconds = sourceUpdateIntervalMinutes * 60 * 1000; // 将分钟转换为毫秒
    setInterval(() => {
    this.fetchAttackerData();
    this.fetchAlertData();
    this.fetchAttackedData();
    }, sourceUpdateIntervalMilliseconds); 
  },
  created() {
    this.fetchAttackerData();
    this.fetchAlertData();
    this.fetchAttackedData();
  },
  computed: {
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
  methods:{
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
    const MM = pad(date.getMonth() + 1); // 月份从0开始，所以要加1
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
    async fetchAttackerData(){
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
      const response_kairos = await axios.get('http://43.138.200.89:8080/kairos/subjects',{
            params:{
              start_time:'2018-04-01 00:00:00',
              end_time:'2018-04-12 00:00:00' ,
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
            危险等级:'可疑'
           });
        }
        for (const subject of response_kairos.data.dangerous_subjects.data) {
            newData.push({
            时间: formatTime(subject.Time),
            类型: subject.SubjectType,
            名称: subject.SubjectName,
            危险等级:'危险'
            });
        }
        this.AttackerTable.data = this.AttackerTable.data.concat(newData);
    },
    async fetchAttackedData(){
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
      const response_kairos = await axios.get('http://43.138.200.89:8080/kairos/objects',{
            params:{
              start_time:'2018-04-01 00:00:00',
              end_time:'2018-04-12 00:00:00' ,
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
            危险等级:'可疑'
           });
        }
        for (const subject of response_kairos.data.dangerous_objects.data) {
            newData_object.push({
            时间: formatTime(subject.Time),
            类型: subject.ObjectType,
            名称: subject.ObjectName,
            危险等级:'危险'
            });
        }
        this.AttackedTable.data = this.AttackedTable.data.concat(newData_object);
    },
    async  fetchAlertData() {
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
        const response_kairos = await axios.get('http://43.138.200.89:8080/kairos/actions',{
            params:{
              start_time:'2018-04-01 00:00:00',
              end_time:'2018-04-12 00:00:00' ,
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
              attackStartTime:formattedStartTime,
              attackEndTime:formattedEndTime,
              securitys:'1,2,3',
            },
            headers: {
              "content-type": "application/json",
            }
          });
          const securityMap = {'1': '低危','2': '中危', '3': '高危'};
          const alarmTypeMap = {'11': '异常登录','10': '暴力破解','8': '可疑操作','14': '动态蜜罐','15': 'webshell','16': '系统后门','13': '反弹shell','9': 'web命令执行','12': '本地提权'};
          const alarms = response.data.body.content;
          this.AlterTable.data = alarms.map(item => ({
          时间: item.hostname,
          主体类型: item.publicIp,
          主体名称: item.publicIp,
          行为: alarmTypeMap[item.alarmType],
          客体类型: item.viewPointVO.attackLocation, // 注意根据实际返回的字段名修改
          客体名称: item.viewPointVO.attackIp, // 同上
        }));
          const newAlterData = [];
          const newDangerData = [];
        for (const subject of response_kairos.data.anomalous_actions.data) {
            newAlterData.push({
              时间:formatTime(subject.Time),
              主体类型: subject.SubjectType,
              主体名称: subject.SubjectName,
              行为: subject.Action,
              客体类型: subject.ObjectType, 
              客体名称: subject.ObjectName,
           });
        }
        for (const subject of response_kairos.data.dangerous_actions.data) {
            newDangerData.push({
              时间:formatTime(subject.Time),
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
  
  }
};
</script>
<style scoped>
.table-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 20px;
}
.table-selector {
  margin-right:20px;
}

.form-select {
  padding: 8px 12px ;
  border: 1px solid #d1b268;
  border-radius: 4px;
  font-size: 14px;
}
.table{
  position: relative;
}
.page_button {
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
</style>
