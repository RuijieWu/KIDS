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
          <!-- 省略部分内容 -->
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
              </p-button>
            </span>
            <div slot="legend">
              <i class="fa fa-circle text-info"></i> 首次出现进程
              <i class="fa fa-circle text-danger"></i> 服务异常
              <i class="fa fa-circle text-warning"></i> 危险
            </div>
          </chart-card>  
          <card :title="warningTable.title" :subTitle="warningTable.subTitle">
            <div slot="raw-content" class="warning_table">
              <paper-table :data="currentPageData" :columns="warningTable.columns">
              </paper-table>
            </div>
          </card>
  
          <div class="page_button">
            <p-button type="info" round @click.native="prevPage" >上一页</p-button>
            <span>第 {{ currentPage }} 页 / 共 {{ totalPages }} 页</span>
            <p-button type="info" round @click.native="nextPage" >下一页</p-button>
          </div>
        </div>
      </div>
    </div>
  </template>