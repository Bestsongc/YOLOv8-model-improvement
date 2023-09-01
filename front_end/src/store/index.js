import Vue from 'vue'
import Vuex from 'vuex'

import cancelAxios from './modules/cancelAxios'

Vue.use(Vuex)

export default new Vuex.Store({
  state: {},
  mutations: {},
  actions: {},
  modules: {
    cancelAxios
  }
})
