<template>
  <div class="container">
    <!-- <div class="logo">
      <img src="../assets/meiyalogo.png" alt=""/>
    </div> -->
    <Menu></Menu>
    <div class="title">
      <span>请选择您想要检测的类型</span>
    </div>
    <div class="content">
      <el-button class="button1" type="info" circle @click="trackByImg">
        <div class="btn-icon">
          <i class="el-icon-camera"></i>
        </div>
        <div class="btn-title">
          <span>图像检测</span>
        </div>
        <div class="btn-content">
          <p>可上传本地文件中的图像进行检测</p>
          <p>支持jpg, png, jpeg等图像格式</p>
        </div>
      </el-button>
      <el-button class="button2" type="info" circle @click="trackByVideo">
        <div class="btn-icon">
          <i class="el-icon-video-camera"></i>
        </div>
        <div class="btn-title">
          <span>视频跟</span>踪
        </div>
        <div class="btn-content">
          <p>可上传本地文件中的视频进行跟踪</p>
          <p>支持mp4, mov等视频格式</p>
        </div>
      </el-button>
      <!-- <el-button class="button3" type="info" circle @click="realtimeTrack">
        <div class="btn-icon">
          <i class="el-icon-odometer"></i>
        </div>
        <div class="btn-title">
          <span>实时跟</span>踪
        </div>
        <div class="btn-content">
          <p>通过设备的摄像头进行实时跟踪</p>
          <p>目前仅支持单类别检测</p>
        </div>
      </el-button> -->
    </div>
  </div>
</template>

<script>
import Menu from '../components/Menu.vue'
export default {
  name: 'Tracking',
  components: {
    Menu
  },
  data () {
    return {}
  },
  methods: {
    // 图片检测
    trackByImg () {
      this.$router.push({path: '/TrackByImg'})
    },
    // 视频检测
    trackByVideo () {
      this.$router.push({path: '/TrackByVideo'})
    },
    // 实时检测
    realtimeTrack () {
      this.$router.push({path: '/RealtimeTracking'})
    }
  }
}
</script>

<style scoped>
.container {
  background: url("../assets/bg5.jpg") no-repeat center;
  background-size: 100% 100%;
}

.container::after {
  z-index: -1;
  position: absolute;
  content: '';
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  background-color: rgba(0, 0, 0, 0.51);
  opacity: 0.3;
}

.logo {
  margin-left: 3%;
  padding-top: 3%;
  width: 8%;
}

.logo img {
  width: 100%;
}

.title {
  margin-top: 1%;
  font-size: 3vw;
  color: #5485c2;
  margin-left: 50px;
}

.title span {
  letter-spacing: 1vw;
}

.content {
  margin-top: 1%;
}

.content button {
  position: relative;
  width: 16%;
  height: 60vh;
  border-radius: 40px;
  transition: all 0.4s;
  margin: 1% 1%;
  /* text-align: center; */
}

.button1{
  background: url("../assets/but1.jpg");
  background-size: 288px 592px;
}

.button2{
  background: url("../assets/but2.jpg");
  background-size: 288px 592px;
}

.button3{
  background: url("../assets/but3.jpg");
  background-size: 288px 592px;
}
.content button:hover {
  width: 18%;
  height: 65vh;
  margin: 0 1%;
  transition: all 0.4s;
}

.content .btn-icon {
  width: 100%;
  position: absolute;
  top: 15%;
  left: 0;
  height: 10%;
  font-size: 4.5vh;
}
.content .btn-title {
  width: 100%;
  position: absolute;
  top: 40%;
  left: 0;
  height: 15%;
  font-size: 3.5vh;
}

.content .btn-title span {
  letter-spacing: .5vw;
}

.content .btn-content {
  width: 100%;
  position: absolute;
  top: 55%;
  left: 0;
  font-size: 2vh;
}

.content .btn-content p{
  margin-top: 5%;
}
</style>
