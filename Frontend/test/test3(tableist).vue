<template>
  <div class="row">
    <div class="table-selector">
      <label for="tableSelect">选择表格:</label>
      <select id="tableSelect" v-model="selectedTable" class="form-select">
        <option value="AgentTable">主机列表</option>
        <option value="AlterTable">警告信息</option>
        <option value="EventTable">事件列表</option>
      </select>
    </div>
    <div class="search-bar">
      <input v-model="searchText" type="text" placeholder="搜索..." class="form-control" />
    </div>
    <div class="col-12" v-if="CurrentTable">
      <card :title="CurrentTable.title" :subTitle="CurrentTable.subTitle" v-if="CurrentTable">
        <div slot="raw-content" class="table">
          <paper-table :data="filteredData" :columns="CurrentTable.columns"></paper-table>
        </div>
      </card>
      <div class="page_button">
        <p-button type="info" round @click.native="handlePrevPage">上一页</p-button>
        <span>第 {{ currentPage }} 页 / 共 {{ totalPages }} 页</span>
        <p-button type="info" round @click.native="handleNextPage">下一页</p-button>
      </div>
    </div>
  </div>
</template>

<script>
import { PaperTable } from "@/components";
const AgentTableColumns = ["ID", "名称", "IP地址", "OS", "Version", "注册日期", "最后上线时间", "状态"];
const AlterTableColumns = ["ID", "时间", "主机", "主机名称", "技术", "策略", "Description", "危害级别"];
const EventTableColumns = ["ID", "时间", "路径", "操作", "RuleDescription", "规则等级"];

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
        data: [
          { ID: "1", 名称: "localhost", IP地址: "127.0.0.1", OS: "OS", Version: "Version", 注册日期: "2024-6-11", 最后上线时间: "2024-6-10", 状态: "Online" },
          // ... (其他数据)
        ],
        options: {
          pageSize: 12,
          currentPage: 1,
        }
      },
      // ... (其他表格数据)
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
      return this.filteredData.length || 0;
    },
    totalPages() {
      return Math.ceil(this.totalItems / this.pageSize) || 1;
    },
    currentPageData() {
      const startIndex = (this.currentPage - 1) * this.pageSize;
      const endIndex = startIndex + this.pageSize;
      return this.filteredData.slice(startIndex, endIndex);
    },
    filteredData() {
      const searchText = this.searchText.trim().toLowerCase();
      if (!searchText) {
        return this.CurrentTable.data;
      }
      return this.CurrentTable.data.filter(item => {
        return Object.values(item).some(value => {
          return String(value).toLowerCase().includes(searchText);
        });
      });
    },
  },
  methods: {
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
  }
};
</script>

<style scoped>
.table-selector {
  margin-bottom: 20px;
  margin-left: 20px;
}

.form-select {
  padding: 8px 12px;
  border: 1px solid #d1b268;
  border-radius: 4px;
  font-size: 14px;
}

.search-bar {
  margin-bottom: 20px;
  margin-left: 20px;
}

.form-control {
  padding: 8px 12px;
  border: 1px solid #d1b268;
  border-radius: 4px;
  font-size: 14px;
  width: 300px;
}

.table {
  position: relative;
}

.page_button {
  position: absolute;
  right: 0;
}
</style>