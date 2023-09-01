// import {SERVER_ADDRESS} from '../static/js/global'
import VueSocketIO from 'vue-socket.io'
import SocketIO from 'socket.io-client'

export const VUE_SOCKET_IO = new VueSocketIO({
  debug: true,
  connection: SocketIO(`${process.env.BASE_URL}/test`)
  // extraHeaders: {'Access-Control-Allow-Origin': '*'}
  // autoConnect: false
})
