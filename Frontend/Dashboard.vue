<template>
  <div>
    <!--Stats cards-->
    <div class="row">
      <div
        class="col-md-6 col-xl-3"
        v-for="stats in statsCards"
        :key="stats.title"
      >
        <stats-card>
          <div
            class="icon-big text-center"
            :class="`icon-${stats.type}`"
            slot="header"
          >
            <i :class="stats.icon"></i>
          </div>
          <div class="numbers" slot="content">
            <p>{{ stats.title }}</p>
            {{ stats.value }}
          </div>
          <div class="stats" slot="footer">
            <i :class="stats.footerIcon"></i> {{ stats.footerText }}
          </div>
        </stats-card>
      </div>
    </div>

    <!--Charts-->
    <div class="row">
  <div class="col-md-8 col-12">
    <chart-card
      title="警报日期分布"
      sub-title="7天内数据"
      :chart-data="usersChart.data"
      :chart-options="usersChart.options"
    >
      <span slot="footer">
        <p-button type="info" round @click.native="refreshData">
          刷新数据
          </p-button>
      </span>
      <div slot="legend">
        <i class="fa fa-circle text-info"></i> 风险
        <i class="fa fa-circle text-danger"></i> 可疑
        <i class="fa fa-circle text-warning"></i> 危险
      </div>
    </chart-card>

    <chart-card
      title="警报时间分布"
      sub-title="24小时内"
      :chart-data="activityChart.data"
      :chart-options="activityChart.options"
    >
      <span slot="footer">
        <p-button type="info" round @click.native="refreshData">
          刷新数据
          </p-button>
      </span>
      <div slot="legend">
        <i class="fa fa-circle text-info"></i> 今天
        <i class="fa fa-circle text-warning"></i> 昨天
      </div>
    </chart-card>
  </div>

  <div class="col-md-4 col-12">
    <chart-card
      title="警报类型统计"
      sub-title="今日内"
      :chart-data="preferencesChart.data"
      chart-type="Pie"
    >
      <span slot="footer">
        <p-button type="info" round @click.native="refreshData">
          刷新数据
          </p-button></span
      >
      <div slot="legend">
        <i class="fa fa-circle text-info"></i> 首次出现进程
        <i class="fa fa-circle text-danger"></i> 服务异常
        <i class="fa fa-circle text-warning"></i> 危险
      </div>
    </chart-card>
    <card :title="warningTable.title" :subTitle="warningTable.subTitle">
      <div slot="raw-content" class="warning_table">
    <paper-table @update:warningTable="handleWarningTableUpdate" :data="currentPageData" :columns="warningTable.columns"  >
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
</template>
<script>

import { StatsCard, ChartCard,PaperTable } from "@/components/index";
import warn_table_child from "@/components/warn_table_child.vue"
import Chartist from "chartist";
export default {
  components: {
    StatsCard,
    ChartCard,
    PaperTable,
    warn_table_child
  },
  data() {
    return {
      usersChartOptions: {},
      statsCards: [
        {
          type: "warning",
          icon: "ti-server",
          title: "主机",
          value: "",
          footerText: "Updated now",
          footerIcon: "ti-reload",
        },
        {
          type: "success",
          icon: "ti-wallet",
          title: "数据总览",
          value: "",
          footerText: "Last day",
          footerIcon: "ti-calendar",
        },
        {
          type: "danger",
          icon: "ti-pulse",
          title: "异常",
          value: "",
          footerText: "In the last hour",
          footerIcon: "ti-timer",
        },
        {
          type: "info",
          icon: "ti-twitter-alt",
          title: "任务",
          value: "",
          footerText: "Updated now",
          footerIcon: "ti-reload",
        },
      ],
      usersChart: {
        data: {
          labels: [
            "5-31",
            "6-1",
            "6-2",
            "6-3",
            "6-4",
            "6-5",
            "6-6",
          ],
          series: [
            [287, 385, 490, 562, 594, 626, 698],
            [67, 152, 193, 240, 387, 435, 535],
            [23, 113, 67, 108, 190, 239, 307],
          ],
        },
        options: {
          low: 0,
          high: 1000,
          showArea: true,
          height: "usersChartOptions",
          axisX: {
            showGrid: false,
          },
          lineSmooth: Chartist.Interpolation.simple({
            divisor: 3,
          }),
          showLine: true,
          showPoint: true,
        },
      },
      activityChart: {
        data: {
          labels: [
          "6:00",
          "9:00", 
          "12:00",
          "15:00",
          "18:00",
          "21:00",
          "24:00",
          "3:00"
          ],
          series: [
            [542, 543, 520, 680, 653, 753, 326, 434],
            [230, 293, 380, 480, 503, 553, 600, 664],
          ],
        },
        options: {
          seriesBarDistance: 10,
          axisX: {
            showGrid: false,
          },
          height: "245px",
        },
      },
      preferencesChart: {
        data: {
          labels: ["62%", "32%", "6%"],
          series: [62, 32, 6],
        },
        options: {},
      },
      warningTable:{
        data:[
      { 时间: "source1", 源地址: "time1", 目的地址: "type1",操作:"1",风险类型:"1" },
      { 时间: "source2", 源地址: "time1", 目的地址: "type1",操作:"1",风险类型:"1" },
      { 时间: "source3", 源地址: "time1", 目的地址: "type1",操作:"1",风险类型:"1" },
      { 时间: "source4", 源地址: "time1", 目的地址: "type1",操作:"1",风险类型:"1" },
      { 时间: "source5", 源地址: "time1", 目的地址: "type1",操作:"1",风险类型:"1" },
      { 时间: "source6", 源地址: "time1", 目的地址: "type1",操作:"1",风险类型:"1" },
      { 时间: "source7", 源地址: "time1", 目的地址: "type1",操作:"1",风险类型:"1" },
      { 时间: "source8", 源地址: "time1", 目的地址: "type1",操作:"1",风险类型:"1" },
      { 时间: "source9", 源地址: "time1", 目的地址: "type1",操作:"1",风险类型:"1" },
      { 时间: "source10", 源地址: "time1", 目的地址: "type1",操作:"1",风险类型:"1" },
      { 时间: "source11", 源地址: "time1", 目的地址: "type1",操作:"1",风险类型:"1" },
    ],
        title: "异常信息",
        subTitle: "",
        columns: ["时间", "源地址", "目的地址", "操作", "风险类型"],
        options:{
          height: "245px",
          pageSize: 6, // 每页显示 8 条数据
          currentPage: 1, // 当前页码
          
        }
      }
    };
  },
  computed: {
    // 当前页码
    currentPage() {
      return this.warningTable.options.currentPage ;
    },
    // 每页条目数
    pageSize() {
      return this.warningTable.options.pageSize || 6;
    },
    // 总条目数
    totalItems() {
      return this.warningTable.data.length || 0;
    },
    // 总页数
    totalPages() {
      return Math.ceil(this.totalItems / this.pageSize);
    },
    // 当前页数据
    currentPageData() {
      const startIndex = (this.currentPage - 1) * this.pageSize;
      const endIndex = startIndex + this.pageSize;
      return this.warningTable.data.slice(startIndex, endIndex);
    },
  },
  mounted() {
    this.setUsersChartHeight();
    window.addEventListener('resize', this.setUsersChartHeight);
    this.fetchChartData(); // 初始请求数据
    this.fetchStatsData();
    this.fetchWarningTableData();
    setInterval(() => {
      this.fetchChartData();
      this.fetchStatsData();
      this.fetchWarningTableData();
    }, 3 * 60 * 60 * 1000); 
  },
  beforeDestroy() {
    window.removeEventListener('resize', this.setUsersChartHeight)
  },
  methods: {
    setUsersChartHeight() {
      const chartWidth = this.$refs.usersChart.$el.offsetWidth
      const aspectRatio = 2 // 或者其他合适的纵横比
      this.usersChartOptions.height = `${chartWidth / aspectRatio}px`
    },
    async fetchChartData() {
      try {
        const response = await fetch('/api/chart-data');
        const chartData = await response.json();

        this.usersChart.data.labels = chartData.usersChart.data.labels;
        this.usersChart.data.series = chartData.usersChart.data.series;
        this.activityChart.data.labels = chartData.activityChart.data.labels;
        this.activityChart.data.series = chartData.activityChart.data.series;
        this.preferencesChart.data.series = chartData.preferencesChart.data.series;

      } catch (error) {
        console.error('表格信息出错:', error);
      }
    },
    async fetchStatsData() {
      try {
        const response = await fetch('/api/stats-value');
        const statsValues = await response.json();
        this.statsCards.forEach(stat => {
          const value = statsValues[stat.title];
            stat.value = value;
        });
      } catch (error) {
        console.error('状态卡片信息出错:', error);
      }
    },
    async fetchWarningTableData() {
      try {
    const {  pageSize } = this.warningTable.options;
    const limit = pageSize * 10; // 假设需要获取 10 页的数据,每页 8 条
    const response = await fetch(`/api/warnings?limit=${limit}`);
    const newData = await response.json();

    // 直接替换旧数据
    this.warningTable.data = newData;
  } catch (error) {
    console.error('获取异常信息出错:', error);
  }
    },
    refreshData() {
      this.fetchChartData();
      this.fetchStatsData();
      this.fetchWarningTableData();
    },
    handlePrevPage() {
      if (this.warningTable.options.currentPage > 1) {
        this.warningTable.options.currentPage--; // 更新当前页码
        
      }
    },
    // 监听子组件触发的下一页事件
    handleNextPage() {
      if (this.warningTable.options.currentPage < this.totalPages) {
        this.warningTable.options.currentPage++; // 更新当前页码
        
      }
    },
  },
};
</script>
<style>
.warning_table {
  position: relative;
}

.page_button {
  position: absolute;
  right: 0;
}
</style>
