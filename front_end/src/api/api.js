// import axios from 'axios'
// import {SERVER_ADDRESS} from '../static/js/global'
//
// export const trackByImg = (data) => {
//   return axios({
//     method: 'post',
//     url: `${SERVER_ADDRESS}/trackByImg`,
//     responseType: 'blob',
//     data: data
//   })
// }
//
// export const trackByVideo = (data) => {
//   return axios({
//     method: 'post',
//     url: `${SERVER_ADDRESS}/trackByVideo`,
//     responseType: 'blob',
//     data: data
//   })
// }
//
// export const cancelTrack = () => {
//   return axios({
//     method: 'get',
//     url: `${SERVER_ADDRESS}/cancelTrack`
//   })
// }
//
// export const realtimeTrackType = (data) => {
//   return axios({
//     method: 'post',
//     url: `${SERVER_ADDRESS}/realtimeTrackType`,
//     data: data
//   })
// }
//
import request from '../util/request'

export const trackByImg = (data) => {
  return request({
    method: 'post',
    url: '/trackByImg',
    responseType: 'blob',
    data: data
  })
}

export const trackByVideo = (data) => {
  return request({
    method: 'post',
    url: '/trackByVideo',
    responseType: 'blob',
    data: data
  })
}

export const realtimeTrackType = (data) => {
  return request({
    method: 'post',
    url: '/realtimeTrackType',
    data: data
  })
}

// export const test = () => {
//   return request({
//     method: 'get',
//     url: '/test'
//   })
// }
