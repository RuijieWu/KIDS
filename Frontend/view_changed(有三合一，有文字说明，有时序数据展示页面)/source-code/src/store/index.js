// store/index.js
import Vue from 'vue'
import Vuex from 'vuex'

Vue.use(Vuex)

// store/index.js
export default new Vuex.Store({
    state: {
      // ... 其他state
      currentSecurityEvent: null
    },
    mutations: {
      // ... 其他mutations
      SET_CURRENT_SECURITY_EVENT(state, event) {
        state.currentSecurityEvent = event;
      }
    },
    actions: {
      // ... 其他actions
      updateCurrentSecurityEvent({ commit }, event) {
        commit('SET_CURRENT_SECURITY_EVENT', event);
      }
    },
    getters: {
      // ... 其他getters
      getCurrentSecurityEvent: state => state.currentSecurityEvent
    }
  });