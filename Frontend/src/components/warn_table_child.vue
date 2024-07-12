<template>
    <div class="pagination">
      <p-button type="info" round @click="handlePrevPage" :disabled="currentPage === 1">上一页</p-button>
      <span>第 {{ currentPage }} 页 / 共 {{ totalPages }} 页</span>
      <p-button type="info" round @click="handleNextPage" :disabled="currentPage === totalPages">下一页</p-button>
    </div>
  </template>
  
  <script>
  import Button from "./index.js";
  export default {
    components: {
    Button
  },
    name: 'warn_table_child',
    props: {
      warningTable: {
        type: Object,
        required: true
      },
      currentPage: {
        type: Number,
        required: true
      },
      totalPages: {
        type: Number,
        required: true
      }
    },
    methods: {
        handlePrevPage() {
      this.$emit('prev-page'); 
    },
    handleNextPage() {
      this.$emit('next-page'); 
    },}}
  </script>
  
  <style scoped>
  .pagination {
    display: flex;
    align-items: center;
    justify-content: flex-end;
    margin-top: 1rem;
  }
  
  .pagination > * {
    margin-left: 0.5rem;
  }
  </style>