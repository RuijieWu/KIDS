import DashboardLayout from "@/layout/dashboard/DashboardLayout.vue";
import NotFound from "@/pages/NotFoundPage.vue";

const routes = [
  {
    path: "/",
    component: DashboardLayout,
    redirect: "/dashboard",
    children: [
      {
        path: "agent",
        name: "agent",
        component: () => import("@/pages/Agent.vue"),
      },
      {
        path: "rule-table",
        name: "rule-table",
        component: () => import("@/pages/Rule.vue"),
      },
      {
        path: "dashboard",
        name: "dashboard",
        component: () => import("@/pages/Dashboard.vue"),
        meta: { requiresLoading: true }
      },
      {
        path: "table-list",
        name: "table-list",
        component: () => import("@/pages/TableList.vue"),
      },
      {
        path:"time",
        name:"time",
        component:() =>import("@/pages/Time.vue"),
      },
      {
        path:"data-ware-house",
        name:"data-ware-house",
        component:()=> import("@/pages/DataWareHouse.vue")
      }, 
      {
        path: "chat",
        name: "chat",
        component: () => import("@/pages/Chat.vue"),
      },
      {
        path: "settings",
        name: "settings",
        component: () => import("@/pages/Settings.vue"),
      },
    ],
  },
  { path: "*", component: NotFound },
];

export default routes;