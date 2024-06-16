<template>
<div>
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
  export default {
    components:{
      RuleChild
    
    },
    data() {
    return {
      searchText: '',
      rules: JSON.parse(localStorage.getItem('rules')) || [
        {ruleName:"sql注入",ruleMessage:"1号",multilineText: '',enable:true,warnLevel:"high",deleted: false}, // 可以传入初始数据
        {ruleName:"55555",ruleMessage:"2号",multilineText: '',enable:true,warnLevel:"low",deleted: false},
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
    clearLocalStorage() {
    localStorage.clear();
    this.rules = []; 
  },
  async deleteRule(ruleId) {
    // 删除对应索引的规则
    alert(ruleId);
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