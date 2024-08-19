<template>
    <div class="main-container">
      <div class="top-row">
        <div class="top-row-left">
          <h4>分析参数</h4>
          <div class="row upload-section">
            <div class="row ml-3">
              <el-tag>上传方式</el-tag>
              <el-select v-model="selectedUploadType" placeholder="选择上传方式" class="ml-1 upload-selector">
                <el-option
                  v-for="item in uploadMethod"
                  :key="item.value"
                  :label="item.label"
                  :value="item.value"
                ></el-option>
              </el-select>
            </div>
            <div class="ml-3 row" v-if="selectedUploadType === 'file'">
              <el-button @click="showUploadDialog" type="primary">选取文件</el-button>
            </div>
            <div class="row align-items-center ml-3" v-else-if="selectedUploadType === 'manual'">
              <div>
                <button class="ml-3 mr-3 mt-0" @click="changeShow">{{ showTextArea ? '收起' : '展开' }}</button>
              </div>
              <div>
                <button class="button upload-button mt-0" @click="sendManualInput" :disabled="!manualInput.trim()">
                  发送数据
                </button>
              </div>
            </div>
          </div>
          <div class="input">
            <textarea
              v-if="showTextArea"
              class="form-control w-100 mt-2"
              style="background-color: gray; color: white"
              rows="3"
              placeholder="输入数据"
              v-model="manualInput"
              @input="submitMultilineText"
            ></textarea>
          </div>
  
          <!-- Dialog for file upload -->
          <el-dialog v-model="dialogVisible" title="选择文件并上传" width="30%" :modal-append-to-body="false">
            
            <el-upload
              class="upload-demo"
              ref="upload"
              action="https://jsonplaceholder.typicode.com/posts/"
              :on-preview="handlePreview"
              :on-remove="handleRemove"
              :file-list="fileList"
              :accept="['.csv', '.xls', '.xlsx']"
              :auto-upload="false"
              list-type="picture-card"
            >
              <el-button slot="trigger" size="small" type="primary" class="mt-0">选取文件</el-button>
              <el-button style="margin-left: 10px;" size="small" type="success" class="mt-0" @click="submitUpload">
                上传到服务器
              </el-button>
            </el-upload>
            <span slot="footer" class="dialog-footer">
              <el-button @click="dialogVisible = false" class="mt-0">取 消</el-button>
            </span>
          </el-dialog>
        </div>
      </div>
    </div>
  </template>
  
  <script>
  export default {
    data() {
      return {
        selectedUploadType: '',
        uploadMethod: [
          { value: 'file', label: '文件上传' },
          { value: 'manual', label: '手动输入' }
        ],
        showTextArea: false,
        manualInput: '',
        dialogVisible: false,
        fileList: []
      };
    },
    methods: {
      showUploadDialog() {
        this.dialogVisible = true;
      },
      handlePreview(file) {
        console.log('handle preview', file);
      },
      handleRemove(file, fileList) {
        console.log('handle remove', file, fileList);
      },
      submitUpload() {
        // 实现上传逻辑
        this.dialogVisible = false;
      },
      changeShow() {
        this.showTextArea = !this.showTextArea;
      },
      sendManualInput() {
        // 实现手动输入发送逻辑
      },
      submitMultilineText() {
        // 处理多行文本输入逻辑
      }
    }
  };
  </script>
  
  <style scoped>
  /* 调整按钮对齐 */
  .el-dialog__footer {
    justify-content: center;
  }
  
  /* 增加页面高度 */
  .main-container {
    height: 100vh; /* 设置页面高度为视口高度 */
  }
  </style>
  