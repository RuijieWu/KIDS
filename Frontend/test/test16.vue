<template>
    <div class="row my-3">
      <div class="col-12">
        <div class="d-flex align-items-center bg-white">
          <div class="flex-grow-1 bg-white">
            <fg-input type="text" class="form-control rule-name-input" placeholder="输入规则名称" v-model="ruleName"  @input="saveInputs">
              </fg-input>
          </div>
          <div class="levelSelect ml-3 mt-3 mb-3">
        <label > 危害等级</label>
        <div >
        <select id="tableSelect" v-model="warnLevel" class="form-select" :class="warnLevelClass" @select="saveInputs">
          <option value="high" class="text-dange">高危</option>
          <option value="medium" class="text-warning">可疑</option>
          <option value="low" class="text-success">低危</option>
        </select>
        </div>
        </div>
        <div class="ruleMessage mt-3 ml-3 mb-3 mr-3"><input type="text" placeholder="输入规则信息" v-model="ruleMessage" @input="saveInputs">
        </div>
        <div class="rule">
            <button class="btn btn-primary ml-3 mr-3" @click="changeShow">{{ showTextArea ? '收起' : '展开' }}</button>
          </div>
          <div class="ml-3">
            <b-form-checkbox switch v-model="enable">启用</b-form-checkbox>
          </div>
          <div class="ml-3 mr-3">
      <b-button variant="success" @click="exportRule">导出规则</b-button>
    </div>
          <button class="btn btn-primary ml-3 mr-3" @click="saveInputs">保存</button>
          <div class="ml-3 mr-3">
            <b-button variant="danger" @click="deleteRule">删除</b-button>
          </div>
        </div>
        <div>
          <textarea v-if="showTextArea" 
          class="form-control"
          style="background-color: gray;
          color: white"
          rows="3" placeholder="输入规则" 
          v-model="multilineText" 
          @input="submitMultilineText">
          </textarea>
        </div>
      </div>
    </div>
  </template>
    
    <script>
    import TextClamp from "./text-clamp.vue"
    import axios from "axios";
    export default {
      name: "rule-child",
      components:{
        TextClamp
      },
      props: {
      rule: {
        type: Object,
        required: true
      },
      ruleId: {
        type: String,
        required: true
      },
      index: {
        type: Number,
        required: true
      },
    },
      data() {
        return {
          ruleName: this.rule.ruleName,
          ruleMessage: this.rule.ruleMessage,
          multilineText: this.rule.multilineText ||'',
          warnLevel:"high",
          enable: this.rule.enable || true,
          showTextArea: false,
          ruleId : this.rule.ruleId
        }
      },
      computed: {
      warnLevelClass() {
        return {
          'text-danger': this.warnLevel === 'high', 
          'text-warning': this.warnLevel === 'medium', 
          'text-success': this.warnLevel === 'low' 
        };
      },
      levelSelectorClass() {
        return {
          'level-selector-danger': this.warnLevel === 'high', 
          'level-selector-warning': this.warnLevel === 'medium', 
          'level-selector-success': this.warnLevel === 'low' 
        };
      }
    },
    mounted() {
      this.loadInputs();
    },
      methods: {
        exportRule() {
      let parsedMultilineText;
      try {
        parsedMultilineText = JSON.parse(this.multilineText);
  
      } catch (error) {
        console.error('无法解析 multilineText:', error);
        alert('无法导出：multilineText 不是有效的 JSON');
        return;
      }
  
      // 创建要导出的对象
      const ruleToExport = {
        ruleName: this.ruleName,
        warnLevel: this.warnLevel,
        ruleMessage:this.ruleMessage,
        multilineText: parsedMultilineText,
        enable: this.rule.enable || true,
        showTextArea: false,
      };
  
      // 使用自定义的格式化函数
      const formattedJsonString = JSON.stringify(ruleToExport, null, 2);
  
      // 创建 Blob
      const blob = new Blob([formattedJsonString], { type: 'application/json' });
  
      // 创建下载链接
      const downloadLink = document.createElement('a');
      downloadLink.href = URL.createObjectURL(blob);
  
      // 设置文件名
      const fileName = `${this.ruleName || 'rule'}.json`;
      downloadLink.download = fileName;
  
      // 触发下载
      document.body.appendChild(downloadLink);
      downloadLink.click();
  
      // 清理
      document.body.removeChild(downloadLink);
      URL.revokeObjectURL(downloadLink.href);
    },
  
        changeShow(){
          this.showTextArea = !this.showTextArea;
          console.log('showTextArea:', this.showTextArea); 
        },
        submitMultilineText() {
          console.log('提交多行文本:', this.multilineText);
          
        },
        async saveInputs() {
        // 当用户按下保存按钮时，将当前页面中所有输入的值保存到本地存储中
        localStorage.setItem(`ruleName_${this.ruleId}`, this.ruleName);
        localStorage.setItem(`warnLevel_${this.ruleId}`,this.warnLevel);
        localStorage.setItem(`ruleMessage_${this.ruleId}`,this.ruleMessage);
        localStorage.setItem(`mutilineText_${this.ruleId}`,this.multilineText);
        localStorage.setItem(`enable_${this.ruleId}`,this.enable);
        const rule = {
        ruleName: this.ruleName,
        warnLevel: this.warnLevel,
        ruleMessage: this.ruleMessage,
        multilineText: this.multilineText,
        enable: this.enable
      };
  
      try {
        // 发送POST请求到后端API
        const response = await axios.post('http://43.138.200.89:8080/blacklist/set-blacklist', rule.multilineText);
        
        console.log('规则成功发送到后端:', response.data);
        // 可以在这里添加成功提示，比如使用 this.$notify 或 alert
      } catch (error) {
        console.error('发送规则到后端时发生错误:', error);
        // 可以在这里添加错误提示
      }
  
      },
        loadInputs() {
        // 在页面加载时从本地存储中获取所有输入的值
        const storedWarnLevel =localStorage.getItem(`warnLevel_${this.ruleId}`);
        const storedRuleName = localStorage.getItem(`ruleName_${this.ruleId}`);
        const storedRuleMessage= localStorage.getItem(`ruleMessage_${this.ruleId}`);
        const storedMutilineText = localStorage.getItem(`mutilineText_${this.ruleId}`);
        const storedEnabled = localStorage.getItem(`enable_${this.ruleId}`);
        this.warnLevel =storedWarnLevel;
        this.ruleName = storedRuleName;
        this.ruleMessage = storedRuleMessage;
        this.multilineText = storedMutilineText;
        this.enable = storedEnabled;
        this.$emit('updateRules');
      },
      deleteRule() {
        
        this.$emit('delete', this.ruleId);
      },
      }
    }
    </script>
    <style>
  .level-selector-danger {
    background-color: red;
  }
  
  .level-selector-warning {
    background-color: yellow;
  }
  
  .level-selector-success {
    background-color: green;
  }
  .rule-name-input{
    font-family:'Franklin Gothic Medium', 'Arial Narrow', Arial, sans-serif;
    font-size: 30px;
  }
  .black-bg {
    background-color: black;
  }
  
  .white-text {
    color: white;
  }
  </style>