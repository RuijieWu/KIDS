import DashboardLayout from "@/layout/dashboard/DashboardLayout.vue";
// GeneralViews
import NotFound from "@/pages/NotFoundPage.vue";

// Admin pages
import Dashboard from "@/pages/Dashboard.vue";
import TableList from "@/pages/TableList.vue";
import Rule from"@/pages/Rule.vue"
import SourceMaps from "../pages/SourceMaps.vue";
import Settings from "@/pages/Settings.vue";
import DataUpload from "@/pages/DataUpload.vue"
const routes = [
  {
    path: "/",
    component: DashboardLayout,
    redirect: "/dashboard",
    children: [
      {
        path: "dashboard",
        name: "dashboard",
        component: Dashboard,
      },
      {
        path: "table-list",
        name: "table-list",
        component: TableList,
      },
      {
        path: "rule-table",
        name: "rule-table",
        component: Rule,   
      },
     {
        path:"source-maps",
        name:"source-maps",
        component: SourceMaps,
     },
     {
        path:"upload-data",
        name:"upload-data",
        component : DataUpload,
     },
     {
      path:"settings",
      name:"settings",
      component:Settings,
     }
    ],
  },
  { path: "*", component: NotFound },
];

/**
 * Asynchronously load view (Webpack Lazy loading compatible)
 * The specified component must be inside the Views folder
 * @param  {string} name  the filename (basename) of the view to load.
function view(name) {
   var res= require('../components/Dashboard/Views/' + name + '.vue');
   return res;
};**/

export default routes;
