import os
from mmdet.apis import init_detector, inference_detector
import numpy as np


def init_yolov3(config_file, checkpoint_file):
    config_file = "vdet/configs/" + config_file  # 'vdet/configs/nas_fpn/retinanet_r50_nasfpn_crop640_50e_coco.py'
    checkpoint_file = "vdet/checkpoints/" + checkpoint_file  # 'vdet/checkpoints/retinanet_r50_nasfpn_crop640_50e_coco-0ad1f644.pth'
    # print(config_file, checkpoint_file)
    model = init_detector(config_file, checkpoint_file, device='cuda:1')
    return model


def inference_yolov3(model, img, det_thresh):
    print(img)
    result = inference_detector(model, img)

    vehicle_res = []
    # print("success",result)
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


def detect(name, config_file="yolo/yolov3_d53_mstrain-608_273e_coco.py",
           checkpoint_file="yolov3_d53_mstrain-608_273e_coco-139f5633.pth",
           det_thresh=0.3):
    model = init_yolov3(config_file, checkpoint_file)
    # print(config_file, checkpoint_file)
    with open('data/' + name + '_result.txt', 'w') as out_file:
        for frame in range(100000):
            frame += 1

            img_path = 'data/images/' + name + ('/%06d.jpg' % frame)
            if not os.path.exists(img_path):
                break
            print(frame)
            result = inference_yolov3(model, img_path, det_thresh=det_thresh)
            for vehicle in result:
                print('%d,%f,%f,%f,%f,%f' % (frame, vehicle[0], vehicle[1], vehicle[2], vehicle[3], vehicle[4]),
                      file=out_file)


def init_detect_model(config_file="yolo/yolov3_d53_mstrain-608_273e_coco.py",
                      checkpoint_file="yolov3_d53_mstrain-608_273e_coco-139f5633.pth"):
    config_file = 'yolo/yolov3_d53_320_273e_coco.py'
    checkpoint_file = 'yolov3_d53_320_273e_coco-421362b6.pth'
    model = init_yolov3(config_file, checkpoint_file)
    return model

def inference_yolov3(model, img, det_thresh):
    result = inference_detector(model, img)

    vehicle_res = []
    # print("success",result)
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

def detect_oneframe(model, img):
    result = inference_yolov3(model, img, det_thresh=0.3)
    return result
