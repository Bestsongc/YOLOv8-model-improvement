<template>
  <div class="container">
    <Menu></Menu>
    <div class="title">
      <span>实时检</span>测
    </div>
    <div class="before-open-camara" v-show="!onCamara">
      <p class="setting-title">基础设置</p>
      <el-form class="info-form">
        <el-form-item label="跟踪类型:">
          <el-select v-model="trackType"
                     placeholder="请选择需要跟踪的类型">
            <el-option-group
              v-for="group in trackTypeList"
              :key="group.label"
              :label="group.label">
              <el-option
                v-for="item in group.options"
                :key="item.value"
                :label="item.label"
                :value="item.value">
              </el-option>
            </el-option-group>
          </el-select>
        </el-form-item>
        <el-form-item label="跟踪设备:">
          <el-select v-model="deviceId"
                     placeholder="请选择跟踪设备">
            <el-option
              v-for="item in deviceList"
              :key="item.deviceId"
              :label="item.label"
              :value="item.deviceId">
            </el-option>
          </el-select>
        </el-form-item>
        <el-form-item label="开启镜像:">
          <el-switch
            v-model="mirror"
            active-color="#262626">
          </el-switch>
        </el-form-item>
      </el-form>
      <el-button type="info" class="setting-btn" @click="openCamara">开启摄像头</el-button>
    </div>
    <div class="after-open-camara" v-show="onCamara">
      <div class="content-left">
        <video id="tracking-video" class="upload-img" autoplay v-show="beforeView||!trackFrame"/>
        <img :src="trackFrame" class="upload-img" alt="" v-show="!beforeView&&trackFrame"/>
      </div>
      <div class="content-right">
        <el-card shadow="always" class="card">
          <div slot="header" class="clearfix">
            <span v-if="!onTrack">跟踪信息</span>
            <el-tag class="tag" type="success" v-else>跟踪中</el-tag>
          </div>
          <el-form class="track-info-form">
            <el-form-item label="跟踪类型:">
              <el-select :value="trackType"
                         disabled>
              </el-select>
            </el-form-item>
            <el-form-item label="跟踪设备:">
              <el-select :value="deviceLabel"
                         disabled>
              </el-select>
            </el-form-item>
            <el-form-item label="开启镜像:">
              <el-switch
                v-model="mirror"
                active-color="#262626">
              </el-switch>
            </el-form-item>
            <el-form-item label="跟踪前视图:" v-show="onTrack">
              <el-switch
                v-model="beforeView"
                active-color="#262626">
              </el-switch>
            </el-form-item>
          </el-form>
          <div v-if="!onTrack">
            <p>如需更换跟踪设备，请关闭摄像头</p>
            <el-button type="primary" round class="track-button" @click="stopCamara">关闭摄像头</el-button>
            <el-button type="primary" round class="track-button" @click="startTrack">开始跟踪</el-button>
          </div>
          <div v-else>
            <el-button type="primary" round class="track-button" @click="endTrack">结束跟踪</el-button>
          </div>
        </el-card>
      </div>
    </div>
  </div>
</template>

<script>
import Menu from '../components/Menu'
import {trackTypeList} from '../static/js/trackTypeList'
import {realtimeTrackType} from '../api/api'

export default {
  name: 'RealtimeTracking',
  components: {
    Menu
  },
  data () {
    return {
      trackFrame: '', // 跟踪完成的帧
      deviceId: '', // 选中设备id
      deviceLabel: '', // 选中设备名字
      deviceList: [], // 设备列表
      onCamara: false, // 是否打开摄像头
      onTrack: false, // 是否开始跟踪
      videoStream: null, // 摄像头视频流
      socketInterval: null, // socket开启时间区间
      beforeView: true, // 查看跟踪前视图
      videoContain: null, // 摄像头视图容器
      mirror: false, // 摄像头镜像
      trackTypeList: trackTypeList,
      trackType: '' // 跟踪类型
    }
  },
  watch: {
    mirror (isMirror) {
      if (isMirror) {
        this.videoContain.style.transform = 'rotateY(180deg)'
      } else {
        this.videoContain.style.transform = 'rotateY(0)'
      }
    }
  },
  sockets: {
    connect () {
      console.log('socket 连接成功')
    },
    image (data) {
      let url = this.arrayBufferToBase64(data)
      this.trackFrame = 'data:image/jpg;base64,' + url
    }
  },
  mounted () {
    this.getDeviceList()
    this.videoContain = document.getElementById('tracking-video')
  },
  beforeDestroy () {
    if (this.videoStream != null) {
      clearInterval(this.socketInterval)
      this.videoStream.getTracks()[0].stop()
    }
  },
  methods: {
    arrayBufferToBase64 (buffer) {
      let binary = ''
      const bytes = new Uint8Array(buffer)
      for (let i = 0; i < bytes.byteLength; i++) {
        binary += String.fromCharCode(bytes[i])
      }
      return window.btoa(binary)
    },
    // 开启摄像头
    openCamara () {
      if (this.trackType === '') {
        this.$message.warning('请选择跟踪类型')
        return
      }
      this.$confirm('即将开启摄像头，是否继续?', '提示', {
        confirmButtonText: '确定',
        cancelButtonText: '取消',
        type: 'info'
      }).then(() => {
        const that = this
        const formData = new FormData()
        formData.append('trackType', this.trackType)
        realtimeTrackType(formData).then(res => {
          if (res) {
            that.onCamara = true
            const constraints = {
              video: {
                deviceId: this.deviceId ? this.deviceId : undefined,
                // width: {min: 640, ideal: 1280, max: 1920},
                // height: {min: 480, ideal: 720, max: 1080}
                width: 256,
                height: 144
              }
            }
            that.deviceList.forEach(device => {
              if (device.deviceId === this.deviceId) {
                that.deviceLabel = device.label
              }
            })
            navigator.mediaDevices.getUserMedia(constraints).then(stream => {
              that.videoContain.srcObject = stream
              that.videoStream = stream
            })
          } else {
            this.$message.error('该类型不支持，请重新选择')
          }
        })
      })
    },
    // 关闭摄像头
    stopCamara () {
      this.videoStream.getTracks()[0].stop()
      this.videoStream = null
      this.videoContain.srcObject = null
      this.onCamara = false
    },
    // 开始跟踪
    startTrack () {
      this.onTrack = true
      this.beforeView = false
      this.socketInterval = setInterval(() => {
        this.$socket.emit('image', this.getFrame(this.videoContain))
      }, 150)
    },
    // 结束跟踪
    endTrack () {
      clearInterval(this.socketInterval)
      this.onTrack = false
      this.beforeView = true
      this.trackFrame = ''
    },
    getFrame (video) {
      const canvas = document.createElement('canvas')
      // console.log(video.videoHeight)
      canvas.width = video.videoWidth
      canvas.height = video.videoHeight
      let ctx = canvas.getContext('2d')
      ctx.clearRect(0, 0, canvas.width, canvas.height)
      if (this.mirror) {
        ctx.translate(canvas.width, 0)
        ctx.scale(-1, 1)
      }
      ctx.drawImage(video, 0, 0)
      // canvas.getContext('2d').drawImage(video, 0, 0)
      // canvas.getContext('2d').scale(-1, 1)
      const data = canvas.toDataURL('image/png')
      console.log(new Date())
      return data
    },
    // 获取设备列表
    getDeviceList () {
      navigator.mediaDevices.enumerateDevices().then(devices => {
        devices.forEach(device => {
          if (device.kind === 'videoinput') {
            this.deviceList.push(device)
          }
        })
        this.deviceId = this.deviceList[0].deviceId ? this.deviceList[0].deviceId : null
      })
    }
  }
}
</script>

<style scoped>
.container {
  background: url("../../assets/realtime_track_bg.jpg") no-repeat center;
}

.title {
  font-size: 3vw;
  color: #cfcece;
}

.title span {
  letter-spacing: 1vw;
}

.before-open-camara {
  margin: 1.5% auto;
  width: 80vh;
  background-color: rgba(255, 255, 255, .7);
  border-radius: 25px;
  padding: 2.5%;
}

.after-open-camara {
  margin: 1.5% auto;
  width: 80vw;
  background-color: rgba(255, 255, 255, .7);
  border-radius: 25px;
  padding: 2.5%;
  display: flex;
}

.before-open-camara .setting-title {
  font-size: 2.8vh;
}

.before-open-camara .info-form {
  margin: 6vh 10vh;
}

.before-open-camara .info-form >>> .el-form-item {
  margin: 4vh 3vh;
}

.before-open-camara .info-form >>> .el-form-item__label {
  font-size: 2.3vh;
}

.before-open-camara .info-form >>> .el-form-item__content {
  text-align: left;
}

.before-open-camara .info-form >>> .el-form-item {
  margin-bottom: 0;
}

.before-open-camara .info-form p {
  margin: 1vh 0;
  font-size: 2vh;
  color: #3c3c3c;
}

.before-open-camara .setting-btn {
  width: 30%;
  margin: 2vh 0 3vh 0;
  font-size: 2vh;
  line-height: 3vh;
}

/*.after-open-camara {*/
/*  display: flex;*/
/*}*/

.after-open-camara .content-left {
  width: 55vw;
  height: 60vh;
  border-radius: 25px;
  background-color: rgba(62, 61, 61, 0.2);
}

.after-open-camara .content-right {
  margin-left: 2%;
  width: 25vw;
  position: relative;
}

.after-open-camara .content-right .card {
  width: 100%;
  bottom: 0;
  height: 60vh;
  border-radius: 25px;
  background-color: rgba(255, 255, 255, .6);
}

.after-open-camara .content-right .card >>> .el-card__header {
  border-bottom: 2px solid #9d9d9d;
  font-size: 2.5vh;
}

.upload-img {
  width: 100%;
  height: 100%;
  border-radius: 25px;
}

.after-open-camara .track-info-form {
  margin: 4.5vh 0;
}

.after-open-camara .track-info-form >>> .el-form-item {
  margin: 3vh .8vw;
}

.after-open-camara .track-info-form >>> .el-select .el-input__inner {
  width: 13vw;
}

.after-open-camara p {
  font-size: 2.2vh;
  margin-bottom: 3.5vh;
}

.after-open-camara .track-button {
  font-size: 1.8vh;
  margin: 0 1.5vh;
  width: 40%;
}

.after-open-camara .content-right .tag {
  width: 10vw;
  height: 5vh;
  line-height: 5vh;
  font-size: 2.5vh;
}
</style>
