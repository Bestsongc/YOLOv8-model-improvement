const cancelAxios = {
  namespace: true,
  state: {
    cancelAxios: null, // 终止axios请求
    reqUrl: 'sss' // 请求url
  },
  mutations: {
    SET_CANCEL_AXIOS (state, cancelAxios) {
      state.cancelAxios = cancelAxios
    },
    SET_REQUEST (state, reqUrl) {
      state.reqUrl = reqUrl
    },
    DEL_REQUEST (state, isCancel) {
      if (isCancel) {
        setTimeout(() => {
          console.log('hhh')
          state.reqUrl = ''
        }, 6000)
      } else {
        state.reqUrl = ''
      }
    }
  },
  actions: {
    setCancelAxios (context, cancelAxios) {
      context.commit('SET_CANCEL_AXIOS', cancelAxios)
    },
    setReqUrl (context, reqUrl) {
      context.commit('SET_REQUEST', reqUrl)
    },
    delReqUrl (context, isCancel) {
      context.commit('DEL_REQUEST', isCancel)
    }
  }
}

export default cancelAxios
