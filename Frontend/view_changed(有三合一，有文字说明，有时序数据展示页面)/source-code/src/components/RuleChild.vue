<template>
  <div class="row my-3">
    <div class="col-12">
      <div class="d-flex align-items-center bg-white">
        <el-tag class="ml-2 mr-2" @click ="changeType">{{ ruleData.ruleType == "black" ? '黑' : '白' }}名单</el-tag>
        <div class="flex-grow-1">
          <fg-input type="text" class="form-control rule-name-input" placeholder="输入规则名称" v-model="ruleData.ruleName" >
            </fg-input>
        </div>
        <div class="levelSelect ml-3 mt-3 mb-3">
      <label > 危害等级</label>
      <div >
      <select id="tableSelect" v-model="ruleData.warnLevel" class="form-select" :class="warnLevelClass" >
        <option value="high" class="text-dange">高危</option>
        <option value="medium" class="text-warning">可疑</option>
        <option value="low" class="text-success">低危</option>
      </select>
      </div>
      </div>
      <div class="ruleMessage mt-3 ml-3 mb-3 mr-3"><el-input type="text" placeholder="输入规则信息" v-model="ruleData.ruleMessage" >
      </el-input>
      </div>
      <div class="rule">
          <button class="btn btn-primary ml-3 mr-3" @click="changeShow">{{ showTextArea ? '收起' : '展开' }}</button>
        </div>
        <div class="ml-3">
          <b-form-checkbox switch v-model="ruleData.enable">启用</b-form-checkbox>
        </div>
        
        <button class="btn btn-primary ml-3 mr-3" @click="saveInputs">保存</button>
        <div class="ml-3 mr-3">
    <b-button variant="success" @click="exportRule">导出</b-button>
        </div>
        <div class="ml-3 mr-3">
          <el-button type="danger" icon="el-icon-delete" circle @click="open"></el-button>
        </div>
      </div>
      <div>
        <textarea v-if="showTextArea" 
        class="form-control"
        style="background-color: gray;
        color: white"
        rows="3" placeholder="输入规则" 
        v-model="displayMultilineText" 
        @input="updateMultilineText($event.target.value)">
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
   console.log(this.ruleId,this.ruleData.id,this.rule.id);
  },


    methods: {

      changeType(){
        let newType = this.ruleData.ruleType == "black" ? "white" :"black";
        this.ruleData.ruleType = newType;
        this.$emit('update:rule', { ...this.ruleData });
      },

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
    
  },

  exportRule() {
  let parsedMultilineText;
  
  if (typeof this.ruleData.multilineText === 'string') {
    try {
      parsedMultilineText = JSON.parse(this.ruleData.multilineText);
    } catch (error) {
      console.error('无法解析 规则内容:', error);
      alert('无法导出:规则内容不是有效的 JSON 格式');
      return;
    }
  } else if (typeof this.ruleData.multilineText === 'object') {
    // 如果已经是对象，直接使用
    parsedMultilineText = this.ruleData.multilineText;
  } else {
    console.error('规则内容格式不正确');
    alert('无法导出:规则内容格式不正确');
    return;
  }

  const ruleToExport = {
    multilineText: parsedMultilineText,
    ruleName: this.ruleData.ruleName,
    warnLevel: this.ruleData.warnLevel,
    ruleMessage: this.ruleData.ruleMessage,
    "deleted":false,
    ruleType: this.ruleData.ruleType,
  };

  console.log(ruleToExport);

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
        /*localStorage.setItem(`ruleName_${this.ruleId}`, this.ruleData.ruleName);
        localStorage.setItem(`warnLevel_${this.ruleId}`, this.ruleData.warnLevel);
        localStorage.setItem(`ruleMessage_${this.ruleId}`, this.ruleData.ruleMessage);
        localStorage.setItem(`multilineText_${this.ruleId}`, this.ruleData.multilineText);
        localStorage.setItem(`enable_${this.ruleId}`, this.ruleData.enable);
        localStorage.setItem(`ruleType_${this.ruleId}`, this.ruleData.ruleType);*/
        console.log("规则",this.ruleData);
        this.$emit('update:rule', { ...this.ruleData });
        console.log("启用",this.ruleData.enable);
        if(this.ruleData.enable){
        try {
          const ruleMessage = {
            ruleName: this.ruleData.ruleName,
            warnLevel: this.ruleData.warnLevel,
            ruleMessage: this.ruleData.ruleMessage,
            deleted: false  
          };
          const dataToSend = `
            {
                {
                  "MultilineText":${JSON.stringify(this.ruleData.multilineText)}
                  "ruleName": ${this.ruleData.ruleName}
                  "warnLevel":${this.ruleData.warnLevel}
                  "ruleMessage":${this.ruleData.ruleMessage}
                  "deleted":fasle
                  "ruleType":${this.ruleData.ruleType}
                  "ruleId":${this.ruleId}
                  }
            }`;
          
          if(this.ruleData.ruleType == "black"){
            const response = await axios.post('http://43.138.200.89:8080/blacklist/set-blacklist', this.ruleData.multilineText);
          }
          else if(this.ruleData.ruleType == "white"){
            const response = await axios.post('http://43.138.200.89:8080/blacklist/set-whitelist', this.ruleData.multilineText);
          }
       } catch (error) {
         console.error('发送规则到后端时发生错误:', error);
        }
      }
  },
      loadInputs() {
      const storedWarnLevel =localStorage.getItem(`warnLevel_${this.ruleId}`);
      const storedRuleName = localStorage.getItem(`ruleName_${this.ruleId}`);
      const storedRuleMessage= localStorage.getItem(`ruleMessage_${this.ruleId}`);
      const storedMutilineText = localStorage.getItem(`mutilineText_${this.ruleId}`);
      const storedEnabled = localStorage.getItem(`enable_${this.ruleId}`);
      const storedRuleType = localStorage.getItem(`ruleType_${this.ruleId}`);
      this.warnLevel =storedWarnLevel;
      this.ruleName = storedRuleName;
      this.ruleMessage = storedRuleMessage;
      this.multilineText = storedMutilineText;
      this.enable = storedEnabled;
      this.ruleType = storedRuleType;
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

.form-select{
  border-radius:10px;
}
</style>