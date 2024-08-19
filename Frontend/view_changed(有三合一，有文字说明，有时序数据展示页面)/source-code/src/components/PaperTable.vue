<template>
  <table class="table" :class="tableClass">
    <thead>
      <slot name="columns">
        <th v-for="column in columns" :key="column.toLowerCase()">{{ column }}</th>
      </slot>
    </thead>
    <tbody>
      <tr v-for="(item, index) in data" :key="index">
        <slot :row="item">
          <td
            v-for="(column, index) in columns"
            :key="index"
          >
            {{ itemValue(item, column) }}
          </td>
        </slot>
      </tr>
    </tbody>
  </table>
</template>
<script>
export default {
  name: "paper-table",
  props: {
    columns: Array,
    data: Array,
    key:{
      type:String,
      default:0,
    },
    type: {
      type: String, // striped | hover
      default: "hover",
    },
    title: {
      type: String,
      default: "",
    },
    subTitle: {
      type: String,
      default: "",
    },
  },
  computed: {
    tableClass() {
      return `table-${this.type}`;
    },
  },
  methods: {
    hasValue(item, column) {
      return item[column.toLowerCase()] !== "undefined";
    },
    itemValue(item, column) {
      return item[column];
    },
  },
  watch: {
  data: {
    handler() {
      this.$forceUpdate();
    },
    deep: true
  }
},
};
</script>
<style></style>
