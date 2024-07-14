/*!

 =========================================================
 * Vue Paper Dashboard - v1.0.1
 =========================================================

 * Product Page: http://www.creative-tim.com/product/paper-dashboard
 * Copyright 2023 Creative Tim (http://www.creative-tim.com)
 * Licensed under MIT (https://github.com/creativetimofficial/paper-dashboard/blob/master/LICENSE.md)

 =========================================================

 * The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

 */

import ElementUI from 'element-ui';
import './plugins/element.js'
import Vue from "vue";
import App from "./App";
import router from "./router/index";
import axios from 'axios';
import PaperDashboard from "./plugins/paperDashboard";
import "vue-notifyjs/themes/default.css";
import { BootstrapVue, IconsPlugin } from 'bootstrap-vue'
import 'element-ui/lib/theme-chalk/index.css';
import 'bootstrap/dist/css/bootstrap.css'
import 'bootstrap-vue/dist/bootstrap-vue.css'
Vue.use(BootstrapVue)

Vue.use(ElementUI);
// 可选: 安装 Bootstrap Vue 图标插件
Vue.use(IconsPlugin)
Vue.use(PaperDashboard);
Vue.prototype.$axios = axios;
/* eslint-disable no-new */
new Vue({
  router,
  render: (h) => h(App),
}).$mount("#app");
