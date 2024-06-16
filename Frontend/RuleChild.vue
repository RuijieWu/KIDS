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
        multilineText: this.rule.multilineText,
        warnLevel:"high",
        enable: this.rule.enable || true,
        showTextArea: false,
      }
    },
    computed: {
    // 计算属性来动态计算 select 元素的类名
    warnLevelClass() {
      return {
        'text-danger': this.warnLevel === 'high', // 如果选项值为 'high'，添加 'text-danger' 类名
        'text-warning': this.warnLevel === 'medium', // 如果选项值为 'medium'，添加 'text-warning' 类名
        'text-success': this.warnLevel === 'low' // 如果选项值为 'low'，添加 'text-info' 类名
      };
    },
    levelSelectorClass() {
      return {
        'level-selector-danger': this.warnLevel === 'high', // 如果选项值为 'high'，添加 'level-selector-danger' 类名
        'level-selector-warning': this.warnLevel === 'medium', // 如果选项值为 'medium'，添加 'level-selector-warning' 类名
        'level-selector-success': this.warnLevel === 'low' // 如果选项值为 'low'，添加 'level-selector-success' 类名
      };
    }
  },
  mounted() {
    this.loadInputs();
  },
    methods: {
      changeShow(){
        this.showTextArea = !this.showTextArea;
        return this.showTextArea;
      },
      submitMultilineText() {
        console.log('提交多行文本:', this.multilineText);
        
      },
      saveInputs() {
      // 当用户按下保存按钮时，将当前页面中所有输入的值保存到本地存储中
      localStorage.setItem(`ruleName_${this.ruleId}`, this.ruleName);
      localStorage.setItem(`warnLevel_${this.ruleId}`,this.warnLevel);
      localStorage.setItem(`ruleMessage_${this.ruleId}`,this.ruleMessage);
      localStorage.setItem(`mutilineText_${this.ruleId}`,this.multilineText);
      localStorage.setItem(`enable_${this.ruleId}`,this.enable);
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