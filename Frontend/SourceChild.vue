<!-- SourceChild.vue -->
<template>
    <div class="source-child">
      <div class="row">
        <div class="col-md-12">
          <div class="d-flex justify-content-between align-items-center mb-3">
            <h4>{{ source.ID }}. {{ source.主机名称 }}</h4>
            <div>
              <span :class="['badge', getBadgeClass(source.危害级别)]">{{ source.危害级别 }}</span>
            </div>
          </div>
          <!-- 其他告警信息 -->
          <div class="mb-3">
            <strong>时间：</strong> {{ source.时间 }}
          </div>
          <div class="mb-3">
            <strong>主机：</strong> {{ source.主机 }}
          </div>
          <div class="mb-3">
            <strong>技术：</strong> {{ source.技术 }}
          </div>
          <div class="mb-3">
            <strong>策略：</strong> {{ source.策略 }}
          </div>
          <div class="mb-3">
            <strong>描述：</strong> {{ source.描述 }}
          </div>
          
          <button class="btn btn-primary" @click="showImageModal">查看相关图片</button>
        </div>
      </div>
  
      <!-- 全屏图片模态框 -->
      <div v-if="showModal" class="modal-overlay" @click="closeImageModal">
        <div class="modal-content" @click.stop>
          <button class="close-btn" @click="closeImageModal">&times;</button>
          <img :src="currentImage" alt="Related Image" class="img-fluid" v-if="currentImage">
          <div v-else>加载中...</div>
          <div class="modal-navigation">
            <button class="btn btn-secondary" @click="prevImage" :disabled="currentImageIndex === 0">上一张</button>
            <button class="btn btn-secondary" @click="nextImage" :disabled="currentImageIndex === images.length - 1">下一张</button>
          </div>
        </div>
      </div>
    </div>
  </template>
  
  <script>
  import axios from 'axios';
  
  export default {
    name: 'SourceChild',
    props: {
      source: {
        type: Object,
        required: true
      }
    },
    data() {
      return {
        showModal: false,
        images: [],
        currentImageIndex: 0,
      };
    },
    computed: {
      currentImage() {
        return this.images[this.currentImageIndex];
      }
    },
    methods: {
        getBadgeClass(level) {
        switch(level.toLowerCase()) {
          case 'high': return 'bg-danger';
          case 'medium': return 'bg-warning';
          case 'low': return 'bg-success';
          default: return 'bg-secondary';
        }
      },
      async showImageModal() {
        this.showModal = true;
        await this.fetchImages();
      },
      closeImageModal() {
        this.showModal = false;
      },
      async fetchImages() {
        try {
          const response = await axios.get(`/api/images/${this.source.ID}`);
          this.images = response.data;
          this.currentImageIndex = 0;
        } catch (error) {
          console.error('Error fetching images:', error);
          this.images = [];
        }
      },
      prevImage() {
        if (this.currentImageIndex > 0) {
          this.currentImageIndex--;
        }
      },
      nextImage() {
        if (this.currentImageIndex < this.images.length - 1) {
          this.currentImageIndex++;
        }
      }
    }
  };
  </script>
  
  <style scoped>
  .source-child {
    background-color: #f8f9fa;
    border-radius: 5px;
    padding: 20px;
    margin-bottom: 20px;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
  }
  
  .modal-overlay {
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background: rgba(0, 0, 0, 0.8);
    display: flex;
    justify-content: center;
    align-items: center;
    z-index: 1000;
  }
  
  .modal-content {
    background: white;
    padding: 20px;
    border-radius: 5px;
    max-width: 80%;
    max-height: 80%;
    overflow: auto;
    position: relative;
  }
  
  .close-btn {
    position: absolute;
    top: 10px;
    right: 10px;
    font-size: 24px;
    cursor: pointer;
    background: none;
    border: none;
    color: #333;
  }
  
  .modal-navigation {
    display: flex;
    justify-content: space-between;
    margin-top: 20px;
  }
  
  .img-fluid {
    max-width: 100%;
    height: auto;
  }
  </style>