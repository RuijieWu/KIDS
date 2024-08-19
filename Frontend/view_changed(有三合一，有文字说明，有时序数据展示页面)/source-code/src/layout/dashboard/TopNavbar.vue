<template>
  <nav class="navbar navbar-expand-lg navbar-light">
    <div class="container-fluid">
      <a class="navbar-brand" href="#">{{ routeName }}</a>
      <el-tooltip class="item custom-tooltip" effect="dark" :content="pageDescription" placement="bottom" raw-content>
        <i class="el-icon-question"></i>
      </el-tooltip>
      <button
        class="navbar-toggler navbar-burger"
        type="button"
        @click="toggleSidebar"
        :aria-expanded="$sidebar.showSidebar"
        aria-label="Toggle navigation"
      >
        <span class="navbar-toggler-bar"></span>
        <span class="navbar-toggler-bar"></span>
        <span class="navbar-toggler-bar"></span>
      </button>
      <div class="collapse navbar-collapse">
        <ul class="navbar-nav ml-auto">
        </ul>
      </div>
    </div>
  </nav>
</template>

<script>
export default {
  data() {
    return {
      activeNotifications: false,
      pageDescriptions: {
        "Dashboard": "  安全态势感知页面展示了引擎输出数据的统计结果，其中四个状态卡片分别代表可疑/危险攻击方数，可疑/危险被攻击方数和可疑/危险行为数。<br>状态卡片的下方使用chartlist中的折线图统计了七日内的可疑/危险行为数量和今日以及昨日不同时间段内的行为数量，使用了echart中的饼图对行为类型，攻击方/被攻击方类型进行了统计与展示，堆叠式柱状图统计了最频繁出现的6个攻击方/被攻击方及其相关的行为。</br>最后使用自定义表格子组件展示了近120条危险/可疑行为的详细信息，上述总体统计数据的展示有利于用户对安全态势有一个总体的认知，便于用户后续使用",
        "Table-list": "  安全事件详细信息页面将引擎输出的四种数据：可疑/危险攻击方，被攻击方，可疑行为，危险行为存储在了四个表格中，通过选择器切换表格。其中攻击方和被攻击方表格存储了对应节点的时间，类型，名称以及危险等级，可疑/危险行为表格存储了对应边的发生时间，源节点类型和名称，目标节点类型和名称以及边对应的行为。安全事件详细信息页面能够帮助用户对安全事件有更具体和详细的认知，以便用户做出更准确的判断来加强防护。",
        "Rule-table": "  数据分析页面允许用户进行自定义黑白名单规则，上传数据文件，修改分析引擎参数等操作。数据分析页面分为三个部分；规则预检，AI模型检测和安全事件输出。规则预检允许用户自定义黑白名单规则并将规则发向后端对原始数据进行过滤，其中每条规则提供文件导入与导出功能，并且提供表格展示规则过滤出来的数据。AI模型分析获取规则预检后的数据，并且允许用户上传数据文件，同时提供用户修改模型参数的功能可以修改包括模型路径，模型名称（决定使用的模型），时间窗口阈值等参数，以及通过与安全态势总览页面相同的组件展示分析结果。安全事件输出部分展示给用户每个可疑时间窗口对应的安全事件卡片，每个安全事件卡片提供查看溯源图（AI模型输出结果，展示可疑时间窗口的行为），查看安全报告（包括一系列统计数据，所有行为的交互式关系图和关键行为的关系图），查看行为表格（时间窗口内的可疑/危险行为的详细信息）以及查看安全建议（包括模板安全建议和大语言模型生成的安全建议）的功能",
        'Agent': "  Agent监控页面展示了agent的相关信息，其中agent指的是部署在不同主机上的数据采集模块。使用饼状图展示了agent的状态分布，并且将Agent信息存储表格中，相关信息包括主机名，IP地址，运行状态，运行时间以及监控路径。",
        "Data-ware-house": "  数据仓库页面，展现了KIDS系统使用的每个数据仓库以及其对应的子节点，子节点信息包括使用率，命中率等，使用饼状图展示了所有数据仓库的总体使用情况。饼状图的右侧列举了系统数据仓库实现的技术。",
        'Chat': "  大语言模型交互页面，提供用户与语言模型交流的功能。",
        "Setting": "  设置页面，用于配置系统参数和选项"
      }
    };
  },
  mounted(){
    console.log('routeName ',this.routeName);
    const style = document.createElement('style');
    style.textContent = `
      .el-tooltip__popper {
        font-size: 18px !important;
      }
    `;
    document.head.appendChild(style);
  },
  computed: {
    routeName() {
      const { name } = this.$route;
      return this.capitalizeFirstLetter(name);
    },
    pageDescription() {
      return this.pageDescriptions[this.routeName] || "页面描述未定义";
    }
  },
  methods: {
    capitalizeFirstLetter(string) {
      return string.charAt(0).toUpperCase() + string.slice(1);
    },
    toggleNotificationDropDown() {
      this.activeNotifications = !this.activeNotifications;
    },
    closeDropDown() {
      this.activeNotifications = false;
    },
    toggleSidebar() {
      this.$sidebar.displaySidebar(!this.$sidebar.showSidebar);
    },
    hideSidebar() {
      this.$sidebar.displaySidebar(false);
    },
  },
};
</script>

<style>
.el-icon-question {
  color: #409EFF;
}

</style>