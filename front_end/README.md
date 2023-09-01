# 多目标检测与跟踪后端项目 multi-tracking_front-end

## Introduction

多目标检测，即指出一张图片上每个目标的位置，并给出其具体类别与置信度。

多目标跟踪(Multi-Object Tracking, MOT)，就是对视频每一帧画面进行多目标检测的基础上，对每个目标分配一个ID，在目标运动过程中，维持每个目标的ID保持不变。

此项目为前端可视化项目，使用Vue.js

实际使用需配合该[项目后端](https://github.com/multi-object-tracking-meiya-ai/multi-tracking_back-end)方可实现完整功能

## Environment

+ Vue 2.x.x
+ Node >= 6.0.0
+ Npm >= 3.0.0

## Build Setup

``` bash
# install dependencies
npm install

# serve with hot reload at localhost:8080
npm run dev

# build for production with minification
npm run build

# build for production and view the bundle analyzer report
npm run build --report
```

## configuration

### 代理服务

需在config/index.js中进行配置，在配置完成后请重新启动项目方可生效

```javascript
proxyTable: {
  '/api' : {
    // target需更换为后端实际局域网内IP，目前仅支持同一网段下的服务
    target: 'http://172.20.10.2:5000',
    ws: true,
    secure: false,
    changeOrigin: true,
    pathRewrite: {
      '^/api': ''
    }
  }
},
```

## Cautions

### 启动

由于项目中存在使用本地摄像头的需求，项目并未上线，因此请使用 http://localhost:8080 进行访问，方可正常使用实时跟踪模块功能

### ffmpeg

部分浏览器不支持H264格式视频解码展示，因此使用视频上传跟踪功能时，可能出现部分视频在上传到网页时无法展示（实际功能正常使用不受影响，跟踪后视频也可正常显示），目前兼容性如下

|        | Safari | Chrome | FireFox | Edge |
| :----: | :----: | :----: | :-----: | :--: |
| 兼容性 |   √    |        |         |  √   |

### 摄像头

获取摄像头设备列表未对所有浏览器进行兼容，因此可能会出现部分浏览器无法选择使用的摄像头（默认调用处于设备列表的第一个摄像头，功能不受影响）推荐使用Chrome浏览器
