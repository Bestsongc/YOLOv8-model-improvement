![a41f5034370fd80647b16d86b52272c](https://github.com/Zwc2003/YOLOv8-model-improvement/assets/126054446/d98e6785-8be7-4837-a0d3-1f8e7bf76d9a)# YOLOv8-model-improvement
## 目录
- [项目基本介绍](#项目基本介绍)
- [启动前后端交互系统](#启动前后端交互系统)
- [如何训练自己的数据集](#如何训练自己的数据集)
## 项目基本介绍
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

3、本项目存放有使用头部数据集训练的改进版模型，并进行模型加速，结果存放于`runs/detect/best-model/weigths`中

   分别是best.pt、best.onnx、best.engine(仅支持GPU)

  
### 完整web端展示界面
web端由Vue和Flask开发，基于目标跟踪实现了两个应用场景的简单功能，两种跟踪器可选（botsort,bytetrack）

![a41f5034370fd80647b16d86b52272c](https://github.com/Zwc2003/YOLOv8-model-improvement/assets/126054446/41913583-5087-4f40-9b08-1914b567ed90)

#### 球员运动轨迹分析
展示每个目标的运动轨迹及速度和加速度

不同目标的轨迹颜色不同

便于赛后复盘比赛和分析战术
#### 逆行检测
检测目标的行进方向是否符合预先设定好的正确方向

在每个逆行目标位置标注其id和轨迹

在视频的左上角集中地展示所有逆行的id
## 启动前后交互系统
### 启动后端
## 如何训练自己的数据集
### 配置环境
需要的环境已在requirements.txt中声明

请在终端输入以下命令(在本项目的根目录下运行)

`pip install -r requirements.txt`

`pip install ultralytics`

### 准备数据集
在根目录下新建一个文件夹，命名为`datasets`

将准备好的yolo格式的数据集放入文件夹中

目录结构应该如下：

>根目录
>>datasets
>>>images
>>>>train
>>>>
>>>>val
>>>
>>>labels
>>>>train
>>>>
>>>>val
### 修改配置文件
1、配置文件基本都在`ultralytics/cfg`的路径下

2、在`ultralytics/cfg/datasets`下设置datasets的路径,项目文件中已有demo.yaml,如果在上面的`准备数据集`的环节中你已经严格按照要求进行,那么demo.yaml的内容几乎不需要修改,若你的数据集放在了其他路径,请复制一份demo.yaml进行修改,根据真实路径模仿下列格式进行修改即可:
```
path: ../datasets/  # dataset root dir
train: images/train  # train images
val: images/val  # val images
test:  # test images (optional)
```
同时,根据你数据集里的目标类型及其对应的id设置下方的`classes`：
```
name:
0 : person
1 : ......
```
### 调参
模型的参数调整集中在了`ultralytics/cfg/default.yaml`的文件中，请根据需要进行调整，其中前两个参数`model`和`data`不需要填写
### 开始训练
打开项目根目录下的`start_train.py`

内容如下：
```
from ultralytics import YOLO

# Load a model
#1、 model = YOLO('yolov8n.yaml')  # build a new model from YAML
#2、 model = YOLO('yolov8n.pt')  # load a pretrained model (recommended for training)
model = YOLO('path/to/the/YAML').load(
             'path/to/the/pre_trained/weights')  # build from YAML and transfer weights

# Train the model
model.train(data='path/to/the/YAML/of/datasets')
```
提供了三种训练模式：(1)从头开始训练一个模型；(2)直接用预训练模型训练自己的数据集；(3)加载预训练权重，用修改后的yaml文件训练自己的数据集

如果你没有修改模型，只是想用YOLOv8训练自己的数据集，那么请使用第2种，不建议用第1种，采用预训练模型可以加快收敛速度，而且效果比较好

如果你修改了模型，那么请使用第3种，因为一般对模型的修改需要在yaml文件中重新声明模型的结构，你一方面通过加载预训练模型来加快收敛提高效果，一方面需要引入修改后的模型的yaml文件

接下来只需要在对应的位置填写`预训练权重/模型的yaml文件/数据集的yaml文件`的路径即可

#### 注意
注意`\`和`/`的区别

预训练权重文件放在`ultralytics/cfg/models/pretrained_models`下

模型的yaml文件放在`ultralytics/cfg/models/v8`下

若需要使用GPU进行训练，请在`default.yaml`中指定`device`参数

### 运行start_train.py文件即可开始训练！







