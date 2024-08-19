<template>
  <div class="wrapper">
    <side-bar>
      <template slot="links">
        <sidebar-link to="/agent" name="数据采集模块状态" icon="ti-control-shuffle" />
        <sidebar-link to="/rule-table" name="二层检测机制" icon="ti-bookmark" />
        <sidebar-link to="/dashboard" name="安全态势感知" icon="ti-panel" />
        <sidebar-link to="/table-list" name="安全事件详细信息 "icon="ti-view-list-alt"/>
        <sidebar-link to="/data-ware-house" name="数据仓库监控" icon="ti-layout-grid4-alt" />
        <sidebar-link to="/chat" name="语言模型交流" icon="ti-pencil-alt" />
        <sidebar-link to="/settings" name="设置" icon="ti-settings" />
      </template>
      <mobile-menu>
        <li class="nav-item">
          <a class="nav-link">
            <i class="ti-panel"></i>
            <p>Stats</p>
          </a>
        </li>
        <drop-down
          class="nav-item"
          title="5 Notifications"
          title-classes="nav-link"
          icon="ti-bell"
        >
          <a class="dropdown-item">Notification 1</a>
          <a class="dropdown-item">Notification 2</a>
          <a class="dropdown-item">Notification 3</a>
          <a class="dropdown-item">Notification 4</a>
          <a class="dropdown-item">Another notification</a>
        </drop-down>
        <li class="nav-item">
          <a class="nav-link">
            <i class="ti-settings"></i>
            <p>Settings</p>
          </a>
        </li>
        <li class="divider"></li>
      </mobile-menu>
    </side-bar>
    <div class="main-panel">
      <top-navbar></top-navbar>

      <dashboard-content @click.native="toggleSidebar"> </dashboard-content>

      
    </div>
  </div>
</template>
<style lang="scss"></style>
<script>
import TopNavbar from "./TopNavbar.vue";
import ContentFooter from "./ContentFooter.vue";
import DashboardContent from "./Content.vue";
import MobileMenu from "./MobileMenu";
export default {
  components: {
    TopNavbar,
    ContentFooter,
    DashboardContent,
    MobileMenu,
  },
  methods: {
    toggleSidebar() {
      if (this.$sidebar.showSidebar) {
        this.$sidebar.displaySidebar(false);
      }
    },
  },
};
</script>
