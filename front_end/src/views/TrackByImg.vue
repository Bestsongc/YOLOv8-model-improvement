<template>
  <div class="container">
    <Menu></Menu>
    <div class="title">
      <span>图像检测</span>
    </div>
    <div class="content">
      <div v-if="!isTracking">
        <el-steps :active="active" align-center class="steps">
            <el-step title="上传检测图片"></el-step>
            <el-step title="选择检测类型"></el-step>
            <el-step title="开始检测"></el-step>
        </el-steps>
        <div class="before-tracking">
        <div class="content-left">
          <div class="before-upload" v-if="!isUpload">
            <el-upload
              action="#"
              drag
              class="uploader"
              list-type="picture"
              :auto-upload="false"
              :show-file-list="false"
              :on-change="imgPreview">
              <div class="uploader-icon">
                <i class="el-icon-upload2"></i>
              </div>
              <div class="uploader-text">请将图像拖到此处或点击上传</div>
            </el-upload>
          </div>
          <div class="after-upload" v-else>
            <img :src="beforeTrackImgUrl" alt="" class="upload-img"/>
            <span class="actions">
              <!-- 放大 -->
              <span class="item">
                <i class="el-icon-zoom-in" @click="beforeImgDialogVisible = true"></i>
              </span>
              <!-- 删除 -->
              <span class="item">
                <i class="el-icon-delete" @click="del"></i>
              </span>
            </span>
          </div>
        </div>
        <div class="content-right">
          <el-card v-if="active === 1" shadow="always" class="card">
            <div slot="header" class="clearfix">
              <span><b>上传检测图片</b></span>
            </div>
            <div v-if="!isUpload" class="step1_before_upload">
              <div class="loading-icon">
                <i class="el-icon-camera-solid" @click="imgCapDialogVisible = true"></i>
              </div>
              <p>未检测到图像上传，请先在 <b>左侧</b> 上传图像</p>
              <p>或点击
                <i class="el-icon-camera-solid"></i>
                进行拍照
              </p>
            </div>
            <div v-else>
              {{ active++ }}
              <!-- <div class="img-info-item">
                {{ '文件名: ' + this.beforeTrackImg.name }}
              </div>
              <div class="img-info-item">
                {{ '类型: ' + this.beforeTrackImg.type }}
              </div>
              <div class="img-info-item">
                {{ '文件大小: ' + this.convertFileSize(this.beforeTrackImg.size) }}
              </div>
              <div class="img-info-step1_2">
                点击 <b>下一步</b> 选择检测类型
              </div>
              <el-button type="primary" round class="img-button" @click="active++">下一步</el-button> -->
            </div>
          </el-card>
          <el-card v-if="active === 2" shadow="always" class="card">
            <div slot="header" class="clearfix">
              <span>选择检测类型</span>
            </div>
            <el-form class="type-form">
              <el-form-item label="检测类型:">
                <el-select v-model="chosenType"
                           placeholder="请选择需要检测的类型"
                           multiple
                           :multiple-limit="3">
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
                <!-- <p>* 请至少选择一种检测类型，最多选择三种检测类型</p> -->
              </el-form-item>
            </el-form>
            <div class="img-tip-step2">
              点击 <b>下一步</b> 开始检测
            </div>
            <!-- <el-button type="primary" round class="img-button" @click="active--">上一步</el-button> -->
            <el-button type="primary" round class="img-button" @click="step2Next">下一步</el-button>
          </el-card>
          <el-card v-if="active === 3" shadow="always" class="card step3">
            <div slot="header" class="clearfix">
              <span>开始检测</span>
            </div>
            <div class="step3">
              <div class="img-info-item">
                {{ '文件名: ' + this.beforeTrackImg.name }}
              </div>
              <div class="img-info-item">
                {{ '类型: ' + this.beforeTrackImg.type }}
              </div>
              <div class="img-info-item">
                {{ '文件大小: ' + this.convertFileSize(this.beforeTrackImg.size) }}
              </div>
              <div class="img-info-item">
                {{ '检测类型: ' + this.chosenType }}
              </div>
            </div>
            <div class="img-info-step3">
              请确认无误后，点击 <b>开始检测</b> 进行检测
            </div>
            <el-button type="primary" round class="img-button" @click="active--">上一步</el-button>
            <el-button type="primary" round class="img-button" @click="uploadImg">开始检测</el-button>
          </el-card>
        </div>
      </div>
      </div>
      <div class="after-tracking" v-else>
        <div class="content-left">
          <div class="before-success-tracking" v-if="!successTrack">
            <i class="el-icon-loading" v-if="trackStatue === 1"></i>
            <i class="el-icon-picture-outline" v-if="trackStatue === 2"></i>
          </div>
          <div class="after-success-tracking" v-else>
            <img :src="afterTrackImgUrl" alt="" class="upload-img"/>
            <span class="actions">
              <!-- 放大 -->
              <span class="item">
                <i class="el-icon-zoom-in" @click="afterImgDialogVisible = true"></i>
              </span>
          </span>
          </div>
        </div>
        <div class="content-right">
          <el-card shadow="always" class="card">
            <div slot="header" class="clearfix">
              <el-tag class="tag" v-if="trackStatue === 0" type="success">检测成功</el-tag>
              <el-tag class="tag" v-if="trackStatue === 1">检测中</el-tag>
              <el-tag class="tag" v-if="trackStatue === 2" type="danger">检测失败</el-tag>
            </div>
            <div class="before-success-tracking" v-if="!successTrack">
              <div class="tracking" v-if="trackStatue === 1">
                <div>
                  <el-progress class="progress" type="circle" :percentage="trackingPercentage"/>
                </div>
                <el-button type="primary" round class="cancel-btn" @click="cancelTrack">取消检测</el-button>
              </div>
              <div class="track-error" v-if="trackStatue === 2">
                <div>
                  <el-progress class="progress" type="circle" :percentage="trackingPercentage" status="exception"/>
                </div>
                <el-button type="primary" round class="cancel-btn" @click="tryAgain">重新检测</el-button>
              </div>
            </div>
            <div class="after-success-tracking" v-else>
              <div class="img-info-item">
                {{ '文件名: ' + this.afterTrackImg.name }}
              </div>
              <div class="img-info-item">
                {{ '类型: ' + this.afterTrackImg.type }}
              </div>
              <div class="img-info-item">
                {{ '文件大小: ' + this.trackFileSize }}
              </div>
              <div class="img-info-item">
                {{ '检测类型: ' + this.chosenType }}
              </div>
              <el-button type="primary" round class="img-info-finish" @click="retrain">重新检测</el-button>
              <el-button type="primary" round class="img-info-finish" @click="downloadImg">下载图像</el-button>
            </div>
          </el-card>
        </div>
      </div>
    </div>
    <!-- 图片显示对话框 -->
    <el-dialog :visible.sync="beforeImgDialogVisible" :modal-append-to-body="false" top="5vh" :show-close="false"
               class="dialog">
      <img :src="beforeTrackImgUrl" alt="" class="dialog-img"/>
    </el-dialog>
    <el-dialog :visible.sync="afterImgDialogVisible" :modal-append-to-body="false" top="5vh" :show-close="false">
      <img :src="afterTrackImgUrl" alt="" class="dialog-img"/>
    </el-dialog>
    <!--    拍照对话框-->
    <div v-show="imgCapDialogVisible" class="img-cap-dialog">
      <div class="img-cap-dialog-content">
        <p class="img-cap-dialog-content-title">拍照上传</p>
        <div v-show="!onCamara" class="before-open-camara">
          <el-form class="info-form">
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
              <el-switch v-model="mirror"
                         active-color="#262626">
              </el-switch>
            </el-form-item>
          </el-form>
          <el-button type="info" class="setting-btn" @click="imgCapDialogVisible = false">取消拍照</el-button>
          <el-button type="info" class="setting-btn" @click="openCamara">开启摄像头</el-button>
        </div>
        <div v-show="onCamara" class="after-open-camara">
          <div class="media-container">
            <video id="tracking-video" autoplay v-show="!trackFrame" class="cap-img"/>
            <img :src="trackFrame" alt="" v-show="trackFrame" class="cap-img"/>
          </div>
          <div class="setting-btn-group">
            <!--            <el-button type="info" class="setting-btn" @click="imgCapDialogVisible = false">取消拍照</el-button>-->
            <el-button type="info" class="setting-btn" @click="stopCamara">关闭摄像头</el-button>
            <el-button type="info" class="setting-btn" v-show="!trackFrame" @click="captureImage">拍照</el-button>
            <el-button type="info" class="setting-btn" v-show="trackFrame" @click="reCapture">重新拍照</el-button>
            <el-button type="info" class="setting-btn" v-show="trackFrame" @click="useCapture">使用图片</el-button>
          </div>
        </div>
      </div>
    </div>
    <!--    <el-carousel v-if="personList !== []" :interval="4000" type="card" height="200px">-->
    <!--      <el-carousel-item v-for="item in personList" :key="item.id">-->
    <!--        <el-image :src="item.img" width="'100%" alt=""></el-image>-->
    <!--      </el-carousel-item>-->
    <!--    </el-carousel>-->
  </div>
</template>

<script>
import Menu from '../components/Menu.vue'
import {trackByImg} from '../api/api'
import {trackTypeList} from '../static/js/trackTypeList'
// import { Upload,ZoomIn,Delete,Camera,Picture,Clock  } from '@element-plus/icons-vue'

export default {
  name: 'TrackByImg',
  components: {
    Menu
    // Upload,
    // ZoomIn,
    // Delete,
    // Camera,
    // Picture,
    // Clock
  },
  data () {
    return {
      active: 1,
      isUpload: false, // false
      beforeTrackImg: '', // 上传图片File
      afterTrackImg: '', // 检测结果图片File
      beforeTrackImgUrl: '', // 上传图片url
      afterTrackImgUrl: '', // 检测结果图片url
      isTracking: false, // 是否开始检测
      beforeImgDialogVisible: false,
      afterImgDialogVisible: false,
      imgCapDialogVisible: false,
      successTrack: false, // false
      trackStatue: 1, // 检测状态（0：成功，1：检测中，2：失败）
      chosenType: [], // 检测类型
      trackTypeList: trackTypeList,
      trackFileSize: '',
      // personList: [], //轨迹追踪所需参数
      trackingPercentage: 0, // 检测进度
      stopProgress: false, // 终止进度条
      deviceList: [], // 设备列表
      deviceId: '', // 选中设备id
      deviceLabel: '', // 选中设备名字
      videoContain: null, // 摄像头视图容器
      mirror: false, // 摄像头镜像
      onCamara: false, // 是否打开摄像头
      videoStream: null, // 摄像头视频流
      trackFrame: '' // 跟踪完成的帧
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
  mounted () {
    this.getDeviceList()
    this.videoContain = document.getElementById('tracking-video')
  },
  beforeDestroy () {
    if (this.$store.state.cancelAxios.cancelAxios !== null) {
      this.$store.state.cancelAxios.cancelAxios()
      this.$store.dispatch('delReqUrl', true)
    }
  },
  methods: {
    // 图片缩略图
    imgPreview (file) {
      console.log(file)
      if (file.raw.type.split('/')[0] === 'image') {
        this.beforeTrackImg = file.raw
        this.beforeTrackImgUrl = URL.createObjectURL(file.raw)
        this.isUpload = true
      } else {
        this.$message({
          type: 'warning',
          message: '请上传正确的图像格式'
        })
      }
    },
    // 删除图片
    del () {
      this.$confirm('此操作将删除该图像, 是否继续?', '提示', {
        confirmButtonText: '确定',
        cancelButtonText: '取消',
        type: 'warning'
      }).then(() => {
        this.beforeTrackImgUrl = ''
        this.isUpload = false
        this.active = 1
        this.$message({
          type: 'success',
          message: '删除成功'
        })
      })
    },
    // 上传图片
    async uploadImg () {
      this.isTracking = true
      this.trackStatue = 1
      this.trackingPercentage = 0
      if (this.beforeTrackImgUrl === '') {
        this.$message.error('请先上传图片或检查图片是否上传成功')
        return
      }
      let formData = new FormData()
      // let fileType = this.beforeTrackImg.raw.type.split('/')[0]
      let suffix = this.beforeTrackImg.name.split('.')
      formData.append('file', this.beforeTrackImg)
      formData.append('trackType', this.chosenType)
      // formData.append('fileType', fileType)
      formData.append('suffix', suffix[suffix.length - 1])
      console.log('开始检测')
      this.refreshProgress(300)
      trackByImg(formData).then(res => {
        this.trackingPercentage = 100
        console.log(res)
        this.afterTrackImgUrl = window.URL.createObjectURL(res.data)
        this.afterTrackImg = new File([res.data], 'result', {type: res.data.type})
        this.trackFileSize = this.convertFileSize(this.afterTrackImg.size)
        console.log(this.afterTrackImg)
        this.successTrack = true
        this.trackStatue = 0
        this.$message({
          message: '检测成功',
          type: 'success'
        })
      }).catch(err => {
        this.trackStatue = 2
        this.stopProgress = true
        this.$message.error(err)
      })
    },
    // 下载图片
    downloadImg () {
      let a = document.createElement('a')
      a.setAttribute('href', this.afterTrackImgUrl)
      a.setAttribute('download', 'result')
      a.click()
    },
    // 重新检测
    retrain () {
      this.$confirm('重新检测将清空本次检测结果, 是否继续?', '提示', {
        confirmButtonText: '确定',
        cancelButtonText: '取消',
        type: 'warning'
      }).then(() => {
        this.isUpload = false
        this.isTracking = false
        this.successTrack = false
        this.beforeTrackImgUrl = ''
        this.afterTrackImgUrl = ''
        this.active = 1
        this.$message({
          type: 'success',
          message: '操作成功'
        })
      })
    },
    // 再试一次
    tryAgain () {
      this.isTracking = false
      this.successTrack = false
      this.active = 3
    },
    // 开启摄像头
    openCamara () {
      this.$confirm('即将开启摄像头，是否继续?', '提示', {
        confirmButtonText: '确定',
        cancelButtonText: '取消',
        type: 'info'
      }).then(() => {
        this.onCamara = true
        const constraints = {
          video: {
            deviceId: this.deviceId ? this.deviceId : undefined,
            width: {min: 640, ideal: 1280, max: 1920},
            height: {min: 480, ideal: 720, max: 1080}
            // width: 256,
            // height: 144
          }
        }
        navigator.mediaDevices.getUserMedia(constraints).then(stream => {
          this.videoContain.srcObject = stream
          this.videoStream = stream
        })
      })
    },
    // 拍照
    captureImage () {
      const canvas = document.createElement('canvas')
      // console.log(video.videoHeight)
      canvas.width = this.videoContain.videoWidth
      canvas.height = this.videoContain.videoHeight
      let ctx = canvas.getContext('2d')
      ctx.clearRect(0, 0, canvas.width, canvas.height)
      if (this.mirror) {
        ctx.translate(canvas.width, 0)
        ctx.scale(-1, 1)
      }
      ctx.drawImage(this.videoContain, 250, 35, 800, 650)
      canvas.toBlob(blob => {
        this.trackFrame = URL.createObjectURL(blob)
        this.beforeTrackImg = new File([blob], 'capture.png', {type: 'image/png'})
      })
    },
    // 重新拍照
    reCapture () {
      this.trackFrame = ''
      this.beforeTrackImg = ''
    },
    // 关闭摄像头
    stopCamara () {
      this.videoStream.getTracks()[0].stop()
      this.videoStream = null
      this.videoContain.srcObject = null
      this.onCamara = false
    },
    // 使用照片
    useCapture () {
      this.beforeTrackImgUrl = this.trackFrame
      this.trackFrame = ''
      this.imgCapDialogVisible = false
      this.isUpload = true
      this.videoStream.getTracks()[0].stop()
      this.videoStream = null
      this.videoContain.srcObject = null
      this.onCamara = false
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
    },
    // 轨迹追踪
    // pursuitTracking () {
    //   pursuitTracking().then(res => {
    //     console.log(res)
    //     const persons = res.data
    //     for (let person in persons) {
    //       this.personList.push({
    //         id: person,
    //         img: window.URL.createObjectURL(new Blob([persons[person]], {type: 'image/jpg'}))
    //       })
    //     }
    //     console.log(this.personList)
    //   })
    // },
    // 文件大小类型换算
    convertFileSize (size) {
      if (size / 1024 > 1) {
        size /= 1024
        if (size / 1024 > 1) {
          size /= 1024
          if (size / 1024 > 1) {
            size /= 1024
            return size.toFixed(2) + ' GB'
          }
          return size.toFixed(2) + ' MB'
        }
        return size.toFixed(2) + ' KB'
      }
      return size.toFixed(2) + ' B'
    },
    // 判断检测类型是否为空
    step2Next () {
      if (this.chosenType.length === 0) {
        this.$message({
          message: '请至少选择一个检测对象',
          type: 'warning'
        })
      } else {
        this.active++
      }
    },
    // 进度条更新
    refreshProgress (timeout) {
      let interval = setInterval(() => {
        this.trackingPercentage++
        if (this.trackingPercentage >= 99 || this.stopProgress) {
          this.stopProgress = false
          clearInterval(interval)
        }
      }, timeout)
    },
    // 取消检测
    cancelTrack () {
      this.$confirm('确认取消检测?', '提示', {
        confirmButtonText: '确定',
        cancelButtonText: '取消',
        type: 'warning'
      }).then(() => {
        this.isTracking = false
        this.active = 3
        this.$store.state.cancelAxios.cancelAxios()
        this.$store.dispatch('delReqUrl', true)
      })
    }
  }
}
</script>

<style scoped>
@import "../static/css/mediaTracking.css";

.title {
  font-size: 3vw;
  color: #5485c2;
}

.title span {
  letter-spacing: 1vw;
}

.container {
  background: url("../assets/bg5.jpg") no-repeat center;
  background-size: 100% 100%;
}

.img-cap {
  border: none;
  background-color: rgba(0, 0, 0, 0);
  font-size: 2.3vh;
  color: rgb(90, 141, 251);
}

.img-cap:hover {
  font-weight: bold;
  color: rgb(73, 108, 214);
}

.img-cap-dialog {
  background-color: rgba(0, 0, 0, .6);
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100vh;
  z-index: 1000;
  float: top;
}

.img-cap-dialog-content {
  margin: 5vh auto;
  width: 80%;
  height: 74vh;
  background-color: rgba(255, 255, 255, .8);
  border-radius: 25px;
  padding: 8vh 3%;
}

.img-cap-dialog-content-title {
  font-size: 3.5vh;
}

.before-open-camara .info-form {
  margin: 10vh 15vw;
}

.before-open-camara .info-form >>> .el-form-item {
  margin: 5vh 5vw;
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

.before-open-camara .setting-btn {
  width: 25%;
  margin: 3vh 2vw;
  font-size: 2vh;
  line-height: 3vh;
}

.media-container {
  margin: 5vh auto;
  width: 75vw;
  height: 60vh;
  border-radius: 25px;
  background-color: rgba(62, 61, 61, 0.2);
}

.cap-img {
  width: 100%;
  height: 100%;
  border-radius: 25px;
}

.after-open-camara .setting-btn {
  width: 20%;
}

.actions {
  position: absolute;
  border-radius: 25px;
  width: 100%;
  height: 100%;
  line-height: 50vh;
  left: 0;
  top: 0;
  cursor: default;
  text-align: center;
  color: #fff;
  opacity: 0;
  font-size: 40px;
  background-color: rgba(0, 0, 0, .5);
  transition: opacity .3s;
}

.actions:hover {
  opacity: 1;
}

.actions:hover span {
  display: inline-block;
}

.actions span {
  display: none;
  margin: 0 10%;
  cursor: pointer;
  font-size: 6vh;
}
</style>
