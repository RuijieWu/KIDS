<template>
    <div>
      <div class="d-flex justify-content-end mr-3">
          <b-button  variant="info" @click="triggerFileInput">导入规则</b-button>
          <input 
            type="file" 
            ref="fileInput" 
            style="display: none" 
            @change="handleFileImport" 
            accept=".json"
          >
        </div>
      <div class="d-flex justify-content-end">
          <button class="btn btn-primary mt-3" @click="addEmptyRule">添加新规则</button>
        </div>
        <div class="col-12 ">
          <rule-child v-for="(rule,index) in rules" 
          :key="index" 
          v-if="!rule.deleted"
          :rule="rule"  
          :rule-id="`rule_${index}`"
          :index = "index"
          @delete="deleteRule"
          @updateRules="updateRules"/>
        </div>
        <div><button @click="clearLocalStorage">清空本地存储</button></div>
        </div>
      </template>
      <script>
      import { ref } from 'vue';
      import RuleChild from "@/components/RuleChild.vue"
      import axios from 'axios';
      export default {
        components:{
          RuleChild
        
        },
        data() {
        return {
          searchText: '',
          rules: JSON.parse(localStorage.getItem('rules')) || [
            {}
          ]
        }
      },
      computed:{
        
      },
      mounted() {
        if (!localStorage.getItem('rules')) {
          this.saveRules(); // 如果本地存储中没有数据，则保存默认数据
        }
      },
      methods: {
        
        triggerFileInput() {
          this.$refs.fileInput.click();
        },
    
       
        handleFileImport(event) {
          const file = event.target.files[0];
          if (file) {
            const reader = new FileReader();
            reader.onload = (e) => {
              try {
                const importedRule = JSON.parse(e.target.result);
                this.importRule(importedRule);
              } catch (error) {
                console.error('无法解析JSON文件:', error);
                alert('无法导入：文件不是有效的JSON格式');
              }
            };
            reader.readAsText(file);
          }
        },
    
    
        importRule(rule) {
          // 验证导入的规则是否包含所需的所有字段
          if (rule.ruleName && rule.warnLevel && rule.ruleMessage && rule.multilineText) {
            // 创建新的规则对象
            const newRule = {
              ruleName: rule.ruleName,
              warnLevel: rule.warnLevel,
              ruleMessage: rule.ruleMessage,
              multilineText: JSON.stringify(rule.multilineText, null, 2),
              enable: true, // 假设新导入的规则默认启用
            };
    
            // 将新规则添加到规则数组中
            this.rules.push(newRule);
    
            // 保存到本地存储
            this.saveRules();
    
            alert('规则导入成功！');
          } else {
            alert('导入失败：JSON文件格式不正确或缺少必要字段');
          }
        },
      
      
      async deleteRule(ruleId) {
        // 删除对应索引的规则
        const ruleIndex = this.rules.findIndex(rule => `rule_${this.rules.indexOf(rule)}` === ruleId);
          if (ruleIndex === -1) {
            console.error('未找到对应的规则');
            return;
          }
          this.rules[ruleIndex].deleted = true;
        this.saveRules();
        // 告知后端规则已删除
        try {
          await this.$axios.delete(`/api/rules/${deletedRule.id}`); // 假设每个规则有一个唯一的id
          console.log('规则删除成功');
        } catch (error) {
          console.error('删除规则时发生错误:', error);
        }
      },
    
    
        saveRules() {
          localStorage.setItem('rules', JSON.stringify(this.rules));
        },
    
    
        addEmptyRule() {
          const newRule = { deleted: false };
          this.rules.push(newRule);
          this.saveRules();
      },
    
    
        loadRules() {
          const storedRules = JSON.parse(localStorage.getItem('rules')) || [];
          this.rules = Array.isArray(storedRules) ? storedRules : [];
        },
    
    
        updateRules() {
          // 更新 rules 数据，重新从本地存储中加载
          this.loadRules();
        },
    
      }
    }
      </script>
      <style>
    .search-bar {
      display: flex;
      align-items: center;
      margin-right: auto;
    }
    </style>