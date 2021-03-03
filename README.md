# traffic-volume

使用 [mmdetection](https://github.com/open-mmlab/mmdetection) 和 [SORT](https://github.com/abewley/sort) 实现的车流量统计demo。

```
- data
    - images 存放从视频截取的帧, 60s+25FPS=1500帧
    - demo.mp4 原视频
    - frame.py 从视频中截取帧 (25FPS)
    - result.txt 调用YOLOv3后的检测结果
- output
    - images 处理后的每帧图像
    - concat.py 将图片拼接成视频
    - output.txt 每帧的tracker信息
    - result.mp4 最终演示视频
- vdet 用mmdetection实现的检测器
    - checkpoints 模型文件
    - configs 配置文件
    - detect.py 输入../data/images的图片，输出检测结果到../data/result.txt
    - inference.py 推理接口
- sort.py SORT模块
```

### 安装

```shell
pip install -r requirements.txt
```

### 使用

```shell
# 获取帧
cd data
python frame.py
```

```shell
# 获取检测结果
cd ../vdet
python detect.py
```

```shell
# 获取车流量统计结果
cd ..
python sort.py --save-frame
```

```shell
# 拼接帧
cd output
python concat.py
```# VelocityOfVehicles
