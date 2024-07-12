<template>
  <div class="row my-3">
    <div class="col-12">
      <div class="d-flex align-items-center bg-white">
        <div class="flex-grow-1 bg-white">
          <fg-input type="text" class="form-control rule-name-input" placeholder="输入规则名称" v-model="ruleData.ruleName"  @input="saveInputs">
            </fg-input>
        </div>
        <div class="levelSelect ml-3 mt-3 mb-3">
      <label > 危害等级</label>
      <div >
      <select id="tableSelect" v-model="ruleData.warnLevel" class="form-select" :class="warnLevelClass" @select="saveInputs">
        <option value="high" class="text-dange">高危</option>
        <option value="medium" class="text-warning">可疑</option>
        <option value="low" class="text-success">低危</option>
      </select>
      </div>
      </div>
      <div class="ruleMessage mt-3 ml-3 mb-3 mr-3"><input type="text" placeholder="输入规则信息" v-model="ruleData.ruleMessage" @input="saveInputs">
      </div>
      <div class="rule">
          <button class="btn btn-primary ml-3 mr-3" @click="changeShow">{{ showTextArea ? '收起' : '展开' }}</button>
        </div>
        <div class="ml-3">
          <b-form-checkbox switch v-model="ruleData.enable">启用</b-form-checkbox>
        </div>
        <div class="ml-3 mr-3">
    <b-button variant="success" @click="exportRule">导出规则</b-button>
  </div>
        <button class="btn btn-primary ml-3 mr-3" @click="saveInputs">保存</button>
        <div class="ml-3 mr-3">
          <b-button variant="danger" @click="open">删除</b-button>
        </div>
      </div>
      <div>
        <textarea v-if="showTextArea" 
        class="form-control"
        style="background-color: gray;
        color: white"
        rows="3" placeholder="输入规则" 
        v-model="displayMultilineText" 
        @input="updateMultilineText">
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
        showTextArea: false,
      }
    },


    computed: {
      ruleData: {
    get() {
      return this.rule;
    },
    set(value) {
      this.$emit('update:rule', value);
    }
  },

  displayMultilineText: {
    get() {
      return typeof this.ruleData.multilineText === 'object' 
        ? JSON.stringify(this.ruleData.multilineText, null, 2) 
        : this.ruleData.multilineText;
    },
    set(value) {
      this.updateMultilineText(value);
    }
  },



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
      open() {
        this.$confirm('确认删除该规则?', '提示', {
          confirmButtonText: '确定',
          cancelButtonText: '取消',
          type: 'warning'
        }).then(() => {
          this.deleteRule();
          this.$message({
            type: 'success',
            message: '删除成功!'
          });
        }).catch(() => {
          this.$message({
            type: 'info',
            message: '已取消删除'
          });          
        });
      },

      updateMultilineText(value) {
    try {
      const parsedValue = JSON.parse(value);
      this.ruleData.multilineText = parsedValue;
    } catch (e) {
      this.ruleData.multilineText = value;
    }
    this.saveInputs();
  },

      exportRule() {
    let parsedMultilineText;
    try {
      parsedMultilineText = JSON.parse(this.ruleData.multilineText);

    } catch (error) {
      console.error('无法解析 multilineText:', error);
      alert('无法导出：multilineText 不是有效的 JSON');
      return;
    }

    const ruleToExport = {
      ruleName: this.ruleData.ruleName,
      warnLevel: this.ruleData.warnLevel,
      ruleMessage:this.ruleData.ruleMessage,
      multilineText: parsedMultilineText,
      enable: this.ruleData.enable || true,
      showTextArea: false,
    };

    const formattedJsonString = JSON.stringify(ruleToExport, null, 2);
    const blob = new Blob([formattedJsonString], { type: 'application/yaml' });
    const downloadLink = document.createElement('a');
    downloadLink.href = URL.createObjectURL(blob);
    const fileName = `${this.ruleData.ruleName || 'rule'}.yaml`;
    downloadLink.download = fileName;
    document.body.appendChild(downloadLink);
    downloadLink.click();
    document.body.removeChild(downloadLink);
    URL.revokeObjectURL(downloadLink.href);
  },

      changeShow(){
        this.showTextArea = !this.showTextArea;
        console.log('showTextArea:', this.showTextArea); 
      },

      async saveInputs() {
        localStorage.setItem(`ruleName_${this.ruleId}`, this.ruleData.ruleName);
        localStorage.setItem(`warnLevel_${this.ruleId}`, this.ruleData.warnLevel);
        localStorage.setItem(`ruleMessage_${this.ruleId}`, this.ruleData.ruleMessage);
        localStorage.setItem(`multilineText_${this.ruleId}`, this.ruleData.multilineText);
        localStorage.setItem(`enable_${this.ruleId}`, this.ruleData.enable);
        
        this.$emit('update:rule', { ...this.ruleData });
       try {
         const response = await axios.post('http://43.138.200.89:8080/blacklist/set-blacklist', this.ruleData.multilineText);
         const responseTest = await axios.get('http://43.138.200.89:8080/blacklist/get-blacklist', {
                params: {
                },
                headers: {
                    'content-type':	'application/yaml',
                }
            });
          console.log('规则成功发送到后端:', response.data);
       } catch (error) {
         console.error('发送规则到后端时发生错误:', error);
        }
      },
      loadInputs() {
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