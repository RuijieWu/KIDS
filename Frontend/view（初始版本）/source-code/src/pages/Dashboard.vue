<template>
  <div>
    
    <div class="row">
    <div
      class="col-md-6 col-xl-3"
      v-for="(stats, index) in statsCards"
      :key="stats.title"
    >
      <stats-card v-loading="stats.loading">
        <div
          class="icon-big text-center"
          :class="`icon-${stats.type}`"
          slot="header"
        >
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

    <!--Charts-->
    <div class="col-12" v-loading="loadingWeek" element-loading-text="加载中...">
    <chart-card
      :key="weekTableKey"
      title="警报日期分布"
      sub-title="7天内数据"
      :chartLibrary="chartist"
      :chart-data="weekTable.data"
      :chart-options="weekTable.options"
    >
      <span slot="footer">
        <p-button type="info" round @click.native="refreshData">
          刷新数据
          </p-button>
      </span>
      <div slot="legend">
        <i class="fa fa-circle text-info"></i> 风险
        <i class="fa fa-circle text-warning"></i> 危险
      </div>
    </chart-card>
  </div>
  <div class="row">
  <!-- 警报时间分布图表 -->
  <div class="col-8" v-loading="loadingDay" element-loading-text="加载中...">
    <chart-card
      :key="dayTableKey"
      title="警报时间分布"
      sub-title="24小时内"
      :chartLibrary="chartist"
      :chart-data="dayTable.data"
      :chart-options="dayTable.options"
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

  <!-- 行为类型统计图表 -->
  <div class="col-4" v-loading="loadingType" element-loading-text="加载中...">
    <el-card class="box-card">
      <div slot="header" class="clearfix">
        <span>今日行为类型统计</span>
      </div>
      <div id="chart" ></div>
    </el-card>
  </div>
</div>
<div class="row" >
  <div class="col-4" v-loading="loadingWeek" element-loading-text="加载中...">
    <chart-card
      title="被攻击方类型"
      chartLibrary="echarts"
      :chartData="objectTypes"
      chart-type="Pie"
    />
  </div>
  <div class="col-8" v-loading="loadingWeek" element-loading-text="加载中...">
  <chart-card
  title="出现最多次被攻击方"
  sub-title=""
  chartLibrary="echarts"
  chartType="Bar"
  :chartData="objectNames"
/>
</div>
</div>
<div class="row">
  <div class="col-4" v-loading="loadingWeek" element-loading-text="加载中...">
    <chart-card
      title="攻击方类型"
      chartLibrary="echarts"
      :chartData="subjectTypes"
      chart-type="Pie"
    />
  </div>
  <div class="col-8" v-loading="loadingWeek" element-loading-text="加载中...">
  <chart-card
  title="出现最多次攻击方"
  sub-title=""
  chartLibrary="echarts"
  chartType="Bar"
  :chartData="subjectNames"
/>
</div>
</div>
<div class="col-12">
    <card :title="warningTable.title" :subTitle="warningTable.subTitle">
      <div slot="raw-content" class="warning_table">
    <paper-table :data="currentPageData" :columns="warningTable.columns"  >
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

import { StatsCard, ChartCard,PaperTable } from "@/components/index";
import * as echarts from 'echarts';
import Chartist from "chartist";
import axios from "axios";
export default {
  components: {
    StatsCard,
    ChartCard,
    PaperTable,
  },
  data() {
    return {
      dayTableKey:0,
      weekTableKey:0,
      echarts: echarts,
      weekTableOptions: {},
      statsCards: [
        {
          type: "warning",
          icon: "ti-server",
          title: "攻击方",
          subTitle:"(可疑/危险)",
          value: "",
          footerIcon: "ti-reload",
          loading:true,
        },
        {
          type: "success",
          icon: "ti-wallet",
          title: "被攻击方",
          subTitle:"(可疑/危险)",
          value: "",
          footerIcon: "ti-calendar",
          loading:true,
        },
        {
          type: "danger",
          icon: "ti-pulse",
          title: "异常行为",
          subTitle:"(可疑/危险)",
          value: "",
          footerIcon: "ti-timer",
          loading:true,
        },
        {
          type: "info",
          icon: "ti-twitter-alt",
          title: "任务",
          subTitle:"()",
          value: "",
          footerIcon: "ti-reload",
          loading:true,
        },
      ],
      weekTable: {
        data: {
          labels: [
            "05-31",
            "06-01",
            "06-02",
            "06-03",
            "06-04",
            "06-05",
            "06-06",
          ],
          series: [
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
          ],
        },
        options: {
          low: 0,
          showArea: true,
          height: "245px",
          axisX: {
            showGrid: false,
          },
          lineSmooth: Chartist.Interpolation.simple({
            divisor: 100,
          }),
          showLine: true,
          showPoint: true,
          animation: {
            duration: 300,
            easing: 'easeOutQuart'
        }
        },
      },
      dayTable: {
        data: {
          labels: [
          "3:00",
          "6:00",
          "9:00", 
          "12:00",
          "15:00",
          "18:00",
          "21:00",
          "24:00",
          ],
          series: [
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
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
      typeGraphy: {
        data: {
          labels: [],
          series: [0, 0, 0],
        },
        options: {},
      },
      warningTable:{
        data:[],
        title: "异常信息",
        subTitle: "",
        columns: ["时间","主体类型","客体名称","危险等级"],
        options:{
          height: "245px",
          pageSize: 6, // 每页显示 8 条数据
          currentPage: 1, // 当前页码
        }
      },
      subjectTypes: [{value:20,name:'netflow'},{value:30,name:'lsof'}],
      objectTypes: [{value:30,name:'netflow'},{value:20,name:'lsof'}],
      subjectNames:[{name:'',value:0,series:[0,0,0,0,0,0,0]}], /* series对应:EVENT_RECVFROM EVENT_SENDTO ,EVENT_EXECUTE ,EVENT_WRITE ,EVENT_OPEN ,EVENT_CLOSE*/
      objectNames:[{name:'',value:0,series:[0,0,0,0,0,0,0]}],
      loadingType:false,
      loadingDay:false,
      loadingWeek:false,
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
    console.log('mounted 钩子被调用'); // 添加调试日志
    const sourceUpdateIntervalMinutes = parseInt(localStorage.getItem('sourceUpdateInterval')) || 2;
    const sourceUpdateIntervalMilliseconds = sourceUpdateIntervalMinutes * 60 * 1000; // 将分钟转换为毫秒
    this.fetchWeekData(); // 初始请求数据
    this.fetchWDayData();
    this.fetchWarningTableData();
    this.fetchStatCardsData();
    this.renderChart();
    window.addEventListener('resize', this.handleResize);
    /*setInterval(() => {
      this.fetchStatCardsData();
      this.fetchWeekData();
      this.fetchWDayData();
      this.fetchWarningTableData();
    }, sourceUpdateIntervalMilliseconds); */
  },

  methods: {
    handleResize() {
  const chartDom = document.getElementById('chart');
  if (chartDom) {
    const chart = this.echarts.getInstanceByDom(chartDom);
    if (chart) {
      chart.resize();
    }
  }
},
    renderChart() {
  this.$nextTick(() => {
    console.log(this.weekTable.data)
    const chartDom = document.getElementById('chart');
    if (chartDom) {
      let chart = echarts.getInstanceByDom(chartDom);
      if (chart) {
        chart.dispose();
      }
      chart = echarts.init(chartDom);
      const option = {
        tooltip: {
          trigger: 'item'
        },
        legend: {
          top: '5%',
          left: 'center'
        },
        series: [
          {
            name: '今日行为类型统计',
            type: 'pie',
            radius: ['40%', '70%'],
            avoidLabelOverlap: false,
            itemStyle: {
              borderRadius: 10,
              borderColor: '#fff',
              borderWidth: 2
            },
            label: {
              show: false,
              position: 'center'
            },
            emphasis: {
              label: {
                show: true,
                fontSize: '40',
                fontWeight: 'bold'
              }
            },
            labelLine: {
              show: false
            },
            data: [
              { value: this.weekTable.data.series[0][6], name: '可疑行为' },
              { value: this.weekTable.data.series[1][6], name: '危险行为' }
            ]
          }
        ]
      };
      chart.setOption(option);
    } else {
      console.error('Chart container not found');
    }
  });
},
    formatDate(date) {
    const pad = (num) => (num < 10 ? '0' + num : num);
    const yyyy = date.getFullYear();
    const MM = pad(date.getMonth() + 1); // 月份从0开始，所以要加1
    const dd = pad(date.getDate());
    const HH = pad(date.getHours());
    const mm = pad(date.getMinutes());
    const ss = pad(date.getSeconds());
    return `${yyyy}-${MM}-${dd} ${HH}:${mm}:${ss}`;
},
async fetchStatCardsData(){
  const response_subjects = await axios.get('http://43.138.200.89:8080/kairos/subjects', {
                params: {
                    limit:"9999",
                    start_time:  '2018-04-01 00:00:00',
                    end_time: '2018-04-12 00:00:00',
                },
                headers: {
                    'content-type':	'application/json',
                    // 根据需要添加授权信息
                }
            });
            this.statsCards[0].value = response_subjects.data.anomalous_subjects.total.toString() +
            '/'+ response_subjects.data.dangerous_subjects.total.toString();
            this.statsCards[0].loading = false
  const response_objects = await axios.get('http://43.138.200.89:8080/kairos/objects', {
                params: {
                    limit:"9999",
                    start_time: '2018-04-01 00:00:00',
                    end_time: '2018-04-12 00:00:00',
                },
                headers: {
                    'content-type':	'application/json',
                    // 根据需要添加授权信息
                }
            });
            this.statsCards[1].value = response_objects.data.anomalous_objects.total.toString()+'/' 
            + response_objects.data.dangerous_objects.total.toString();
            this.statsCards[1].loading = false
            const response_statics = await axios.get('http://43.138.200.89:8080/kairos/aberration-statics', {
                params: {
                    start_time: '2018-04-01 00:00:00',
                    end_time: '2018-04-12 00:00:00',
                },
                headers: {
                    'content-type':	'application/json',
                    // 根据需要添加授权信息
                }
            });                    
  
  
  this.statsCards[3].value = response_statics.data.total.toString();
  this.statsCards[3].loading = false
},
async fetchWDayData() {
  this.loadingDay = true;
    try {
        const now = new Date();
        const todayStart = new Date('2018-04-12T00:00:00');
        const yesterdayStart = new Date('2018-04-11T00:00:00');

        // 定义今天的时间段
        const timeIntervals = [
            "00:00","3:00", "6:00", "9:00", "12:00", "15:00", "18:00", "21:00"
        ];
        // 格式化今天时间段的日期
        /*const formattedDatesToday = timeIntervals.map(interval => {
            const [hours, minutes] = interval.split(':');
            const startTime = new Date(todayStart);
            startTime.setHours(parseInt(hours), parseInt(minutes), 0, 0);
            return this.formatDate(startTime); // 使用辅助函数格式化日期
        });*/
        // 格式化昨天时间段的日期
        /*const formattedDatesYesterday = timeIntervals.map(interval => {
            const [hours, minutes] = interval.split(':');
            const startTime = new Date(yesterdayStart);
            startTime.setHours(parseInt(hours), parseInt(minutes), 0, 0);
            return this.formatDate(startTime); // 使用辅助函数格式化日期
        });*/
        const formattedDatesToday = ['2018-04-10 00:00:00','2018-04-10 03:00:00','2018-04-10 06:00:00','2018-04-10 09:00:00','2018-04-10 12:00:00',
          '2018-04-10 15:00:00','2018-04-10 18:00:00','2018-04-10 21:00:00',
        ];
        const formattedDatesYesterday = ['2018-04-09 00:00:00','2018-04-09 03:00:00','2018-04-09 06:00:00','2018-04-09 09:00:00','2018-04-09 12:00:00',
          '2018-04-09 15:00:00','2018-04-09 18:00:00','2018-04-09 21:00:00',
        ];
        // 获取今天各个时间段的数据
        const todayDataPromises = formattedDatesToday.map(async (startTime, index) => {
            const endTime = index === formattedDatesToday.length - 1 ? this.formatDate(new Date('2018-04-13T00:00:00')) : formattedDatesToday[index + 1];
            const response = await axios.get('http://43.138.200.89:8080/alarm/alarm/list', {
                params: {
                    page: '1',
                    pageSize: '1',
                    alarmTypes: '11,10,8,14,15,16,13,9,12',
                    attackStartTime: startTime,
                    attackEndTime: endTime,
                    securitys:'1,2,3'
                },
                headers: {
                    'content-type':	'application/json',
                    // 根据需要添加授权信息
                }
            });
            const response_kairos = await axios.get('http://43.138.200.89:8080/kairos/actions', {
                params: {
                    page: '1',
                    pageSize: '1',
                    start_time: startTime,
                    end_time: endTime,
                },
                headers: {
                    'content-type':	'application/json',
                    // 根据需要添加授权信息
                }
            });
            const total_today = response.data.body.total+response_kairos.data.anomalous_actions.total+response_kairos.data.dangerous_actions.total;
            return total_today;

        });

        // 获取昨天各个时间段的数据
        const yesterdayDataPromises = formattedDatesYesterday.map(async (startTime, index) => {
            const endTime = index === formattedDatesYesterday.length - 1 ? formattedDatesToday[0] : formattedDatesYesterday[index + 1];
            const response = await axios.get('http://43.138.200.89:8080/alarm/alarm/list', {
                params: {
                    page: '1',
                    pageSize: '1',
                    alarmTypes: '11,10,8,14,15,16,13,9,12',
                    attackStartTime: startTime,
                    attackEndTime: endTime,
                    securitys:'1,2,3'
                },
                headers: {
                    'content-type':	'application/json',
                    // 根据需要添加授权信息
                }
            });
            const response_kairos = await axios.get('http://43.138.200.89:8080/kairos/actions', {
                params: {
                    page: '1',
                    pageSize: '1',
                    start_time: startTime,
                    end_time: endTime,
                },
                headers: {
                    'content-type':	'application/json',
                    // 根据需要添加授权信息
                }
            });
            const total_yesterday = response.data.body.total+response_kairos.data.anomalous_actions.total+response_kairos.data.dangerous_actions.total
            return total_yesterday;
        });

        // 等待所有请求返回结果
        const todayCounts = await Promise.all(todayDataPromises);
        const yesterdayCounts = await Promise.all(yesterdayDataPromises);

        // 更新 dayTable.data 中的数据
        this.dayTable.data.series[0] = todayCounts;
        this.dayTable.data.series[1] = yesterdayCounts;
        console.log("每日数据",this.dayTable.data);
    } catch (error) {
        console.error('获取活动数据失败：', error);
        this.loadingDay = false;
    }finally{
      this.loadingDay = false;
      this.dayTableKey+=1;
    }
},
/*async fetchWDayData() {
  try {
    const formatDate = (date) => {
      const pad = (num) => (num < 10 ? '0' + num : num); // 小于 10 的数字前面加 0
      const yyyy = date.getFullYear(); // 获取年份
      const MM = pad(date.getMonth() + 1); // 获取月份，月份从 0 开始，所以要加 1
      const dd = pad(date.getDate()); // 获取日期
      const HH = pad(date.getHours()); // 获取小时
      const mm = pad(date.getMinutes()); // 获取分钟
      const ss = pad(date.getSeconds()); // 获取秒
      return `${yyyy}-${MM}-${dd} ${HH}:${mm}:${ss}`; // 拼接成字符串并返回
    };

    const timeIntervals = [
      "3:00", "6:00", "9:00", "12:00", "15:00", "18:00", "21:00", "24:00"
    ];
    const getDataForDay = async (date) => {
      const formattedDates = timeIntervals.map(interval => {
        const [hours, minutes] = interval.split(':');
        const startTime = new Date(date);
        startTime.setHours(parseInt(hours), parseInt(minutes), 0, 0);
        return formatDate(startTime);
      });
      const data = [];
      for (let i = 0; i < formattedDates.length; i++) {
        const startTime = formattedDates[i];
        const endTime = i === formattedDates.length - 1 ? formatDate(new Date(date.getTime() + 24 * 60 * 60 * 1000)) : formattedDates[i + 1];
        const response = await axios.get('http://43.138.200.89:8080/alarm/alarm/list', {
          params: {
           
            start_time: startTime,
            end_time: endTime,
          },
          headers: {
            'content-type': 'application/json',
          }
        });

        data.push(response.anomalous_actions.total+response.dangerous_actions.total);
      }
      return data;
    };

    const date1 = new Date('2018-04-12');
    const date2 = new Date('2018-04-11');

    const todayCounts = await getDataForDay(date1);
    const yesterdayCounts = await getDataForDay(date2);

    this.dayTable.data.series = [todayCounts, yesterdayCounts];
  } catch (error) {
    console.error('获取活动数据失败：', error);
  }
},*/


async fetchWeekData() {
  this.loadingWeek = true;
    try {
        const today = new Date();
        const formattedDates_kairos = ["2018-04-01 00:00:00","2018-04-02 00:00:00","2018-04-03 00:00:00","2018-04-04 00:00:00",
            "2018-04-05 00:00:00","2018-04-06 00:00:00","2018-04-07 00:00:00","2018-04-08 00:00:00","2018-04-09 00:00:00","2018-04-10 00:00:00",
            "2018-04-11 00:00:00","2018-04-12 00:00:00"
        ];
        const formattedDates_kairos_day = ["04-01","04-02","04-03","04-04",
            "04-05","04-06","04-07","04-08","04-09","04-10",
            "04-11","04-12"
        ];
        today.setHours(0, 0, 0, 0); // 将时间设为当天的开始

        for (let i = 0; i < 7; i++) {
            const day = new Date(today);
            day.setDate(today.getDate() - i);
            const formattedDate = (`0${day.getMonth() + 1}`).slice(-2) + '-' + (`0${day.getDate()}`).slice(-2); // 格式化日期为 MM-DD
            this.weekTable.data.labels[6 - i] = formattedDates_kairos_day[9 - i];
            const date_open = [];
            date_open[6-i] = formattedDate;
        }
        
        const formattedDates = []; // 用于存储格式化后的日期时间
        var total_actions_anomalous = 0;
        var total_action_danger = 0;

        const subjectTypes = {};
        const objectTypes = {};
        const subjectNames = {};
        const objectNames = {};
        const actionTypes = ["EVENT_RECVFROM", "EVENT_SENDTO", "EVENT_EXECUTE", "EVENT_WRITE","EVENT_READ", "EVENT_OPEN", "EVENT_CLOSE"];

        for (let j = 1; j <= 3; j++) {
            for (let i = 0; i < 7; i++) {
                const k = i * j;
                const attackEndTime = formattedDates[10 - i];
                const attackStartTime = formattedDates[9 - i];
                const response_open = await axios.get('http://43.138.200.89:8080/alarm/alarm/list', {
                    params: {
                        page: '1',
                        pageSize: '1',
                        alarmTypes: '11,10,8,14,15,16,13,9,12',
                        attackStartTime: attackStartTime,
                        attackEndTime: attackEndTime,
                        securitys: j.toString(),
                        sign:'userchart'
                    },
                    headers: {
                        "content-type": "application/json",
                    }
                });

                if(j == 3){
                    const attackEndTime_kairos = formattedDates_kairos[10 - i];
                    const attackStartTime_kairos = formattedDates_kairos[9 - i];
                    console.log(attackStartTime_kairos,attackEndTime_kairos)
                    const response_kairos = await axios.get('http://43.138.200.89:8080/kairos/actions', {
                        params: {
                            page: '1',
                            pageSize: '1',
                            start_time: attackStartTime_kairos,
                            end_time: attackEndTime_kairos,
                            sign:"userchart"
                        },
                        headers: {
                            "content-type": "application/json",
                        }
                    });
                    this.weekTable.data.series[0][6-i] += response_kairos.data.anomalous_actions.total;
                    this.weekTable.data.series[1][6-i] += response_kairos.data.dangerous_actions.total;
                    console.log("危险行为数",attackStartTime_kairos,attackEndTime_kairos,this.weekTable.data.series[1][6-i])
                    total_actions_anomalous += response_kairos.data.anomalous_actions.total;
                    total_action_danger += response_kairos.data.dangerous_actions.total;
                    
                        const allActions = response_kairos.data.anomalous_actions.data.concat(response_kairos.data.dangerous_actions.data);
                        allActions.forEach(action => {
                            const subjectName = action.SubjectName;
                            const objectName = action.ObjectName;
                            const actionTypeIndex = actionTypes.indexOf(action.Action);
                            if (!subjectNames[subjectName]) {
                                subjectNames[subjectName] = { name: subjectName, value: 0, series: [0, 0, 0, 0, 0, 0,0] };
                            }
                            subjectNames[subjectName].value += 1;
                            subjectNames[subjectName].series[actionTypeIndex] += 1;
                            if (!objectNames[objectName]) {
                                objectNames[objectName] = { name: objectName, value: 0, series: [0, 0, 0, 0, 0, 0,0] };
                            }
                            objectNames[objectName].value += 1;
                            objectNames[objectName].series[actionTypeIndex] += 1;

                            subjectTypes[action.SubjectType] = (subjectTypes[action.SubjectType] || 0) + 1;
                            objectTypes[action.ObjectType] = (objectTypes[action.ObjectType] || 0) + 1;
                        });
                    
                } else {
                    this.weekTable.data.series[0][6-i] += response_open.data.body.total;
                    total_actions_anomalous += response_open.data.body.total;
                }
            }
            
        }
        this.statsCards[2].value = total_actions_anomalous.toString()+'/'+total_action_danger.toString();
        this.statsCards[2].loading = false
        this.loadingType = false;
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
        this.renderChart()
    } catch (error) {
        this.error = '获取图表数据失败';
    } finally {
      this.weekTableKey += 1;
      this.loadingWeek = false;
    }
},

    async fetchWarningTableData() { 
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
      const now = new Date();
      const todayStart = new Date(now);
      todayStart.setHours(0, 0, 0, 0); // 设置时间为今天的开始（午夜）
      const response_kairos = await axios.get('http://43.138.200.89:8080/kairos/actions', {
        params: {
          start_time: "2018-04-01 00:00:00",
          end_time: "2018-04-12 00:00:00",
        },
        headers: {
      ' content-type':'application/json',
        }
      });
          const newAlterData = [];
          const newDangerData = [];
          for (let i = 0; i < response_kairos.data.anomalous_actions.data.length&&i<60; i++) {
            const subject = response_kairos.data.anomalous_actions.data[i];
            newAlterData.push({
              时间:formatTime(subject.Time),
              主体类型: subject.SubjectType,
              主体名称: subject.SubjectName,
              行为: subject.Action,
              客体类型: subject.ObjectType, 
              客体名称: subject.ObjectName,
              危险等级:'可疑'
           });
        };
        for (let i = 0; i < response_kairos.data.dangerous_actions.data.length&&i<60; i++) {
          const subject = response_kairos.data.anomalous_actions.data[i];
            newDangerData.push({
              时间:formatTime(subject.Time),
              主体类型: subject.SubjectType,
              主体名称: subject.SubjectName,
              行为: subject.Action,
              客体类型: subject.ObjectType, 
              客体名称: subject.ObjectName,
              危险等级:'危险'
            });
        };
        this.warningTable.data = this.warningTable.data.concat(newDangerData);
        this.warningTable.data = this.warningTable.data.concat(newAlterData);
        
},
    refreshData() {
      this.fetchWeekData();
      this.fetchWDayData();
      this.fetchWarningTableData();
    },
    handlePrevPage() {
      if (this.warningTable.options.currentPage > 1) {
        this.warningTable.options.currentPage--; // 更新当前页码
        
      }
    },
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
.row {
  display: flex;
}
.col-8, .col-4 {
  display: flex;
  flex-direction: column;
}

.chart-card, .el-card {
  flex: 1;
  display: flex;
  flex-direction: column;
}

.el-card__body {
  flex: 1;
  display: flex;
  flex-direction: column;
}

#chart {
  flex: 1;

}
</style>
