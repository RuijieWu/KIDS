<template>
    <div class="row">
      <div class="col-12">
        <card :title="table1.title" :subTitle="table1.subTitle">
          <div slot="raw-content" class="table-responsive">
            <paper-table :data="table1.data" :columns="table1.columns">
            </paper-table>
          </div>
        </card>
      </div>
  
      <div class="col-12">
        <card class="card-plain">
          <div class="table-full-width table-responsive">
            <paper-table
              type="hover"
              :title="table2.title"
              :sub-title="table2.subTitle"
              :data="table2.data"
              :columns="table2.columns"
            >
            </paper-table>
          </div>
        </card>
      </div>
    </div>
  </template>
  
  <script>
  import { PaperTable } from "@/components";
  
  export default {
    components: {
      PaperTable,
    },
    data() {
      return {
        table1: {
          title: "Stripped Table",
          subTitle: "Here is a subtitle for this table",
          columns: ["Id", "Name", "Salary", "Country", "City"],
          data: [],
        },
        table2: {
          title: "Table on Plain Background",
          subTitle: "Here is a subtitle for this table",
          columns: ["Id", "Name", "Salary", "Country", "City"],
          data: [],
        },
      };
    },
    created() {
      this.fetchTableData();
    },
    methods: {
      // 模拟从钩子请求数据的方法
      fetchTableData() {
        // 这里可以是你的数据请求逻辑
        // 假设请求成功后更新表格数据
        // 例如：
        setTimeout(() => {
          this.table1.data = [
            { id: 1, name: "John Doe", salary: "$50,000", country: "USA", city: "New York" },
            { id: 2, name: "Jane Smith", salary: "$45,000", country: "Canada", city: "Toronto" },
            // 添加更多数据
          ];
          this.table2.data = [
            { id: 3, name: "Alice Johnson", salary: "$55,000", country: "UK", city: "London" },
            { id: 4, name: "Bob Brown", salary: "$60,000", country: "Australia", city: "Sydney" },
            // 添加更多数据
          ];
        }, 1000); // 假设数据请求需要1秒钟完成
      },
    },
  };
  </script>
  
  <style></style>