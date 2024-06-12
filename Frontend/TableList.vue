<template>
  <div class="row">
    <div class="col-12">
    <div class="table-header">
    <div class="table-selector">
      <label for="tableSelect">选择表格:</label>
      <select id="tableSelect" v-model="selectedTable" class="form-select">
        <option value="AgentTable">主机列表</option>
        <option value="AlterTable">警告信息</option>
        <option value="EventTable">事件列表</option>
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
const AgentTableColumns = ["ID", "名称", "IP地址", "OS", "Version","注册日期","最后上线时间","状态"];
const AlterTableColumns = ["ID","时间","主机","主机名称","技术","策略","Description","危害级别"]
const EventTableColumns = ["ID","时间","路径","操作","RuleDescription","规则等级"]
export default {
  components: {
    PaperTable,
  },
  data() {
    return {
      selectedTable: 'AgentTable',
      searchText: '',
      AgentTable: {
        title: "主机列表",
        subTitle: "",
        columns: [...AgentTableColumns],
        data: [{ID:"1",名称:"localhost",IP地址:"127.0.0.1",OS:"OS",Version:"Version",注册日期:"2024-6-11",最后上线时间:"2024-6-10",状态:"Online"},
        {ID:"2",名称:"localhost",IP地址:"127.0.0.1",OS:"OS",Version:"Version",注册日期:"2024-6-11",最后上线时间:"2024-6-10",状态:"Online"},
        {ID:"3",名称:"localhost",IP地址:"127.0.0.1",OS:"OS",Version:"Version",注册日期:"2024-6-11",最后上线时间:"2024-6-10",状态:"Online"},
        {ID:"4",名称:"localhost",IP地址:"127.0.0.1",OS:"OS",Version:"Version",注册日期:"2024-6-11",最后上线时间:"2024-6-10",状态:"Online"},
        {ID:"5",名称:"localhost",IP地址:"127.0.0.1",OS:"OS",Version:"Version",注册日期:"2024-6-11",最后上线时间:"2024-6-10",状态:"Online"},
        {ID:"6",名称:"localhost",IP地址:"127.0.0.1",OS:"OS",Version:"Version",注册日期:"2024-6-11",最后上线时间:"2024-6-10",状态:"Online"},
        {ID:"7",名称:"localhost",IP地址:"127.0.0.1",OS:"OS",Version:"Version",注册日期:"2024-6-11",最后上线时间:"2024-6-10",状态:"Online"},
        {ID:"8",名称:"localhost",IP地址:"127.0.0.1",OS:"OS",Version:"Version",注册日期:"2024-6-11",最后上线时间:"2024-6-10",状态:"Online"},
        {ID:"8",名称:"localhost",IP地址:"127.0.0.1",OS:"OS",Version:"Version",注册日期:"2024-6-11",最后上线时间:"2024-6-10",状态:"Online"},
        {ID:"8",名称:"localhost",IP地址:"127.0.0.1",OS:"OS",Version:"Version",注册日期:"2024-6-11",最后上线时间:"2024-6-10",状态:"Online"},
        {ID:"8",名称:"localhost",IP地址:"127.0.0.1",OS:"OS",Version:"Version",注册日期:"2024-6-11",最后上线时间:"2024-6-10",状态:"Online"},
        {ID:"8",名称:"localhost",IP地址:"127.0.0.1",OS:"OS",Version:"Version",注册日期:"2024-6-11",最后上线时间:"2024-6-10",状态:"Online"},
        {ID:"9",名称:"localhost",IP地址:"127.0.0.1",OS:"OS",Version:"Version",注册日期:"2024-6-11",最后上线时间:"2024-6-10",状态:"Online"},
        {ID:"10",名称:"localhost",IP地址:"127.0.0.1",OS:"OS",Version:"Version",注册日期:"2024-6-11",最后上线时间:"2024-6-10",状态:"Online"} 
      ],
        options:{
          pageSize: 12, 
          currentPage: 1, 
        }
      },
      AlterTable: {
        title: "警告信息",
        subTitle: "",
        columns: [...AlterTableColumns],
        data: [],
        options:{
          pageSize: 12, 
          currentPage: 1, 
        }
      },
      EventTable:{
        title: "事件列表",
        subTitle: "",
        columns: [...EventTableColumns],
        data: [],
        options:{
          pageSize: 12, 
          currentPage: 1, 
        }
      }
    };
  },
  created() {
    this.fetchAgentData();
    this.fetchAlertData();
    this.fetchEventData();
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
      return this.CurrentTable.data.length || 0;
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
      const searchText = this.searchText.trim().toLowerCase();
      if (!searchText) {
        return this.currentPageData;
      }
      return this.currentPageData.filter(item => {
        return Object.values(item).some(value => {
          return String(value).toLowerCase().includes(searchText);
        });
      });
    },
},
  methods:{
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
    fetchAgentData() {
      axios
        .get("/api/agents")
        .then((response) => {
          this.AgentTable.data = response.data;
        })
        .catch((error) => {
          console.error("Error fetching agent data:", error);
        });
    },
    fetchAlertData() {
      axios
        .get("/api/alerts")
        .then((response) => {
          this.AlterTable.data = response.data;
        })
        .catch((error) => {
          console.error("Error fetching alert data:", error);
        });
    },
    fetchEventData() {
      axios
        .get("/api/events")
        .then((response) => {
          this.EventTable.data = response.data;
        })
        .catch((error) => {
          console.error("Error fetching event data:", error);
        });
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
