import os
from mmdet.apis import init_detector, inference_detector
import numpy as np


def init_yolov3():
    config_file = 'vdet/configs/yolo/yolov3_d53_320_273e_coco.py'
    checkpoint_file = 'vdet/checkpoints/yolov3_d53_320_273e_coco-421362b6.pth'
    model = init_detector(config_file, checkpoint_file, device='cuda:0')
    return model


def inference_yolov3(model, img_path, det_thresh=0.5):
    result = inference_detector(model, img_path)

    vehicle_res = []

    for car in result[2]:
        if car[4] > det_thresh:
            vehicle_res.append(car.tolist())

    for bus in result[5]:
        if bus[4] > det_thresh:
            vehicle_res.append(bus.tolist())

    for truck in result[7]:
        if truck[4] > det_thresh:
            vehicle_res.append(truck.tolist())

    return vehicle_res


def detect(name):
    model = init_yolov3()

    import time
    temp = time.time()

    with open('data/' + name + '_result.txt', 'w') as out_file:
        for frame in range(1500):
            frame += 1

            print(frame)

            img_path = 'data/images/' + name + ('/%06d.jpg' % frame)
            if not os.path.exists(img_path):
                break
            result = inference_yolov3(model, img_path, det_thresh=0.1)
            for vehicle in result:
                print('%d,%f,%f,%f,%f,%f' % (frame, vehicle[0], vehicle[1], vehicle[2], vehicle[3], vehicle[4]),
                      file=out_file)

    temp = time.time() - temp
