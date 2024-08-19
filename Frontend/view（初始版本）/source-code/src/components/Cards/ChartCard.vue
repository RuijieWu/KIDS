<template>
  <card>
    <template slot="header">
      <h4 v-if="$slots.title || title" class="card-title">
        <slot name="title">{{ title }}</slot>
      </h4>
      <p class="card-category">
        <slot name="subTitle">{{ subTitle }}</slot>
      </p>
    </template>
    <div>
      <div :id="chartId" class="chart-container"></div>
      <div class="footer">
        <div class="chart-legend">
          <slot name="legend"></slot>
        </div>
        <hr />
        <div class="stats">
          <slot name="footer"></slot>
        </div>
        <div class="pull-right"></div>
      </div>
    </div>
  </card>
</template>

<script>
import Card from "./Card.vue";

export default {
  name: "chart-card",
  components: {
    Card,
  },
  props: {
    key:{
      type:Number,
      default:0,
    },
    chartLibrary: {
      type: String,
      default: 'chartist',
      validator: value => ['chartist', 'echarts'].includes(value)
    },
    title: {
      type: String,
      default: "",
    },
    subTitle: {
      type: String,
      default: "",
    },
    chartType: {
      type: String,
      default: "Line", // Line | Pie | Bar
    },
    chartData: {
      type: [Object,Array],
      required: true,
    },
    chartOptions: {
      type: Object,
      default: () => ({}),
    },
  },
  data() {
    return {
      chartId: "chart-" + this._uid,
      chart: null,
    };
  },
  methods: {
    initChart() {
      if (this.chartLibrary === 'chartist') {
        this.initChartist();
      } else if (this.chartLibrary === 'echarts') {
        this.initECharts();
      }
    },
    initChartist() {
  import("chartist").then((Chartist) => {
    let ChartistLib = Chartist.default || Chartist;
    this.$nextTick(() => {
      const chartIdQuery = `#${this.chartId}`;
      if (this.chart) {
        this.chart.detach();
      }

      this.chart = new ChartistLib[this.chartType](chartIdQuery, this.chartData, this.chartOptions);

      // 监听 Chartist 的 'draw' 事件，在每个点上添加标签
      this.chart.on('draw', (data) => {
        if (data.type === 'point') {
          const value = data.value.y;

          // 创建标签元素
          const label = new Chartist.Svg('text');
          label.text(value);
          label.addClass('ct-label');
          label.attr({
            x: data.x,
            y: data.y - 10, // 调整位置
            'text-anchor': 'middle'
          });

          // 将标签添加到点上
          data.element.parent().append(label);
        }
      });
    });
  });
},

    initECharts() {
      if (!this.chart) {
        this.chart = this.$echarts.init(document.getElementById(this.chartId));
      }
      const option = this.createEChartsOption();
      this.chart.setOption(option);
    },
    createEChartsOption() {
      const actionTypes = ["接收", "发送", "执行", "写入","读取", "打开", "关闭"];
      if (this.chartType === "Pie") {
        return {
          backgroundColor: '#ffffff',  
          tooltip: {
            trigger: 'item'
          },
          series: [
            {
              name: 'Types',
              type: 'pie',
              radius: '55%',
              center: ['50%', '50%'],
              data: Array.isArray(this.chartData) ? this.chartData : Object.entries(this.chartData).map(([name, value]) => ({ name, value })),
              roseType: 'radius',
              label: {
                color: '#333' 
              },
              labelLine: {
                lineStyle: {
                  color: '#999' 
                },
                smooth: 0.2,
                length: 10,
                length2: 20
              },
              itemStyle: {
                color: '#3498db',  
                shadowBlur: 200,
                shadowColor: 'rgba(0, 0, 0, 0.5)'
              },
              animationType: 'scale',
              animationEasing: 'elasticOut',
              animationDelay: function (idx) {
                return Math.random() * 200;
              }
            }
          ]
        };
      } else if (this.chartType === "Bar") {
        return {
          tooltip: {
            trigger: 'axis',
            axisPointer: {
              type: 'shadow'
            }
          },
          legend: {},
          grid: {
            left: '3%',
            right: '4%',
            bottom: '3%',
            containLabel: true
          },
          xAxis: {
            type: 'value'
          },
          yAxis: {
            type: 'category',
            data: this.chartData.map(item => item.name)  // 以subjectNames的name作为Y轴
          },
          series: this.chartData[0].series.map((_, index) => ({
            name: actionTypes[index],
            type: 'bar',
            stack: 'total',
            label: {
              show: true
            },
            emphasis: {
              focus: 'series'
            },
            data: this.chartData.map(item => item.series[index])  // 以subjectNames的series对应的数据作为值
          }))
        };
      }
    },
    updateChart() {
      if (this.chartLibrary === 'chartist' && this.chart) {
        this.chart.update(this.chartData, this.chartOptions);
      } else if (this.chartLibrary === 'echarts' && this.chart) {
        const option = this.createEChartsOption();
        this.chart.setOption(option);
      }
    },
    handleResize() {
    if (this.chart) {
      if (this.chartLibrary === 'echarts') {
        this.chart.resize();
      } else if (this.chartLibrary === 'chartist') {
        this.chart.update();
      }
    }
  },
  },
  watch: {
  chartData: {
    handler() {
      this.$nextTick(() => {
        if (this.chartLibrary === 'chartist') {
          this.initChartist();
        } else if (this.chartLibrary === 'echarts') {
          this.initECharts();
        }
      });
    },
    deep: true,
    immediate: true
  },
  chartOptions: {
    handler() {
      this.$nextTick(() => {
        if (this.chartLibrary === 'chartist') {
          this.initChartist();
        } else if (this.chartLibrary === 'echarts') {
          this.initECharts();
        }
      });
    },
    deep: true
  }
},
mounted() {
  this.$nextTick(() => {
    this.initChart();
    window.addEventListener('resize', this.handleResize);
  });
},

beforeDestroy() {
  if (this.chart) {
    if (this.chartLibrary === 'echarts') {
      this.chart.dispose();
    } else if (this.chartLibrary === 'chartist') {
      this.chart.detach();
    }
    this.chart = null;
  }
  window.removeEventListener('resize', this.handleResize);
},
};
</script>

<style scoped>
.chart-container {
  height: 300px;
}
</style>
