# YOLOv8-model-improvement
## 目录
- [项目基本介绍](#项目基本介绍)
- [no](#no)
## 项目基本结束
本项目对著名的YOLOv8模型进行了多角度的改进优化，并在特定数据集上实现了精度的上涨

采用改进版本的YOLOv8模型与两个表现优良的跟踪器结合(botsort和bytetrack)

基于Vue和Flask开发了一个目标跟踪算法展示平台,平台提供了图像检测和视频跟踪两种功能

视频跟踪实现了在两种应用场景下简单功能（逆行检测和球员轨迹分析）
### YOLOv8目标检测组合优化改进（成功涨点）:

#### 1、添加GAM注意力机制；
添加部位为backbone：

![image](https://github.com/Zwc2003/YOLOv8-model-improvement/assets/126054446/63bd8895-5085-4010-bbea-98d37de0ed92)



#### 2、添加小目标检测头；
新增检测4X4以上目标的检测头，提高对小目标的检测能力
![image](https://github.com/Zwc2003/YOLOv8-model-improvement/assets/126054446/8c71bdcf-2688-4e47-86c3-4f43e4b04e86)


#### 3、替换为Wise_IoU损失函数
本项目中已将YOLOv8内置的CIoU替换为Wise-IoU

### 实验数据集
yolo格式的人体头部数据集(主要由教室等场所的摄像头拍摄获得)

train:3475

val:868

### 历次有效提升的实验结果

![image](https://github.com/Zwc2003/YOLOv8-model-improvement/assets/126054446/76f631ab-e49f-403f-b59d-c9ca851945b8)

### 注意！！！
1、项目中保留有其他修改实验的配置文件，如添加其他注意力机制的YAML文件，单独添加小目标检测头的YAML文件，可根据需要进行训练

2、需要注意的是，本项目使用的边界盒回归损失函数已经修改为Wise-IoU

3、本项目存放有使用头部数据集训练的改进版模型，并进行模型加速，结果存放于runs/detect/best-model/weigths中

   分别是best.pt、best.onnx、best.engine(仅支持GPU)

  
### 完整web端展示界面
web端基于目标跟踪实现了两个应用场景的简单功能，两种跟踪器可选（botsort,bytetrack）
#### 球员运动轨迹分析
展示每个目标的运动轨迹及速度和加速度

不同目标的轨迹颜色不同

便于赛后复盘比赛和分析战术
#### 逆行检测
检测目标的行进方向是否符合预先设定好的正确方向

在每个逆行目标位置标注其id和轨迹

在视频的左上角集中地展示所有逆行的id

## no
shoxuainsxahicnlasck

俺是菜鸟四年
萨的参数才能看



