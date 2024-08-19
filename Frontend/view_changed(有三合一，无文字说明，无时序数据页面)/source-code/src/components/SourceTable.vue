<!-- SourceTable.vue -->
<template>
  <div class="source-table">
    <h3>{{ title }}</h3>
    <div class="row">
      <div v-for="source in paginatedSources" :key="source.ID" class="col-md-6">
        <source-child :source="source"></source-child>
      </div>
    </div>
    <div class="d-flex justify-content-center mt-4">
      <button class="btn btn-primary mr-2" @click="prevPage" :disabled="currentPage === 1">上一页</button>
      <span class="align-self-center mx-3">第 {{ currentPage }} 页 / 共 {{ totalPages }} 页</span>
      <button class="btn btn-primary" @click="nextPage" :disabled="currentPage === totalPages">下一页</button>
    </div>
  </div>
</template>

<script>
import SourceChild from './SourceChild.vue';

export default {
  name: 'source-table',
  components: {
    SourceChild
  },
  props: {
    sources: {
      type: Array,
      required: true
    },
    title: {
      type: String,
      default: '警告信息'
    }
  },
  data() {
    return {
      currentPage: 1,
      pageSize: 4,
      displayedSources: [],
    };
  },
  computed: {
    
    totalPages() {
      return Math.ceil(this.sources.length / this.pageSize);
    },
    paginatedSources() {
      const start = (this.currentPage - 1) * this.pageSize;
      const end = start + this.pageSize;
      return this.sources.slice(start, end);
    }
  },
  methods: {

    async prevPage() {
      if (this.currentPage > 1) {
        this.currentPage--;

      }
    },
    async nextPage() {
      if (this.currentPage < this.totalPages) {
        this.currentPage++;

      }
    }
  },
  mounted() {

  }
}
</script>