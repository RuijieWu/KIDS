 
 <template>
    <div class="text-clamp">
      <div class="text" :style="{height}">
        <span v-if="isVisible" class="btn" @click="toggle">{{isExpand ? '收起' : '... 展开'}}</span>
        <div ref="textRef" :style="commonStyle">
          <slot />
        </div>
      </div>
    </div>
  </template>
   
  <script>
  export default {
    name: "text-clmp",
    props: {
      fontSize: {
        type: Number,
        default: 14
      },
      lines: {
        type: Number,
        default: 1
      },
      lineHeight: {
        type: Number,
        default: 20
      },
      selectors: {
        type: String,
        default: ""
      }
    },
    data () {
      return {
        isExpand: false,
        isVisible: false,
        textHeight: 0
      }
    },
    computed: {
      height () {
        if (this.isExpand) {
          return this.$refs.textRef.clientHeight + 'px';
        } else {
          return Math.min((this.lines * this.lineHeight), this.textHeight) + 'px';
        }
      },
      commonStyle () {
        return {
          lineHeight: this.lineHeight + 'px',
          fontSize: this.fontSize + 'px',
        }
      }
    },
    mounted () {
      this.init();
      // 监听插槽变化
      const observer = new MutationObserver((mutations) => {
        mutations.forEach((mutation) => {
          if (mutation.type === "characterData") {
            this.init();
          }
        });
      });
      observer.observe(this.$refs.textRef, {
        characterData: true,
        subtree: true,
        childList: true
      });
    },
    methods: {
      init () {
        this.isExpand = false;
        this.textHeight = (this.$refs && this.$refs.textRef && this.$refs.textRef.clientHeight) || 0;
        this.isVisible = this.textHeight > this.lines * this.lineHeight;
      },
      toggle () {
        this.isExpand = !this.isExpand;
        if (!this.isExpand && this.selectors) {
          const initEl = document.querySelector(this.selectors);
          setTimeout(() => {
            initEl.scrollIntoView({
              behavior: 'smooth',
              block: 'start',
              inline: 'center'
            });
          }, 97)
        }
      }
    }
  }
  </script>
   
  <style lang="scss" scoped>
  .text-clamp {
    display: flex;
    overflow: hidden;
  }
  .text {
    font-size: 20px;
    transition: 0.3s height;
  }
  .text::before {
    content: "";
    height: calc(100% - 20px);
    float: right;
  }
  .btn {
    float: right;
    clear: both;
    font-size: 12px;
    line-height: 14px;
    padding: 2px 6px;
    background: #1890ff;
    border-radius: 2px;
    color: #fff;
    cursor: pointer;
  }
  </style>