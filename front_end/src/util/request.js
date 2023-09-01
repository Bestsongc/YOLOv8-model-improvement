import axios from 'axios'
import store from '../store'

// 创建 axios 实例
const request = axios.create({
  // API 请求的默认前缀
  baseURL: process.env.BASE_URL
  // timeout: 5000 // 请求超时时间
})

// 正在进行中的请求列表
// let reqList = []

/**
 * 阻止重复请求
 * @param {array} reqList - 请求缓存列表
 * @param {string} url - 当前请求地址
 * @param {function} cancel - 请求中断函数
 * @param {string} errorMessage - 请求中断时需要显示的错误信息
 */
const stopRepeatRequest = function (url, cancel, errorMessage) {
  const errorMsg = errorMessage || ''
  if (store.state.cancelAxios.reqUrl === url) {
    cancel(errorMsg)
  }
  // for (let i = 0; i < store.state.reqUrl.length; i++) {
  //   if (reqList[i] === url) {
  //     cancel(errorMsg)
  //     return
  //   }
  // }
  // reqList.push(url)
}

/**
 * 允许某个请求可以继续进行
 * @param {array} reqList 全部请求列表
 * @param {string} url 请求地址
 */
// const allowRequest = function (reqList, url) {
//   for (let i = 0; i < reqList.length; i++) {
//     if (reqList[i] === url) {
//       reqList.splice(i, 1)
//       break
//     }
//   }
// }

// 请求拦截器
request.interceptors.request.use(
  config => {
    let cancel
    // 设置cancelToken对象
    config.cancelToken = new axios.CancelToken(c => {
      cancel = c
    })
    // 阻止重复请求。当上个请求未完成时，相同的请求不会进行
    stopRepeatRequest(config.url, cancel, '请求过于频繁，请稍后再试')
    store.dispatch('setCancelAxios', cancel)
    store.dispatch('setReqUrl', config.url)
    return config
  }, err => Promise.reject(err)
)
//
// 响应拦截器
request.interceptors.response.use(
  response => {
    // 增加延迟，相同请求不得在短时间内重复发送
    // // setTimeout(() => {
    store.dispatch('delReqUrl', false)
    // allowRequest(reqList, response.config.url)
    // // }, 1000)
    store.dispatch('setCancelAxios', null)
    return response
  }, err => Promise.reject(err)
)

export default request
