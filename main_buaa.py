from __future__ import print_function

import sys
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
from model.camera_calibration import get_parameters
from model.cut_data import cutdata
from model.detect_updated import detect
from model.deep_sort.deep_sort import DeepSort
from model.tracker import Sort, Iou
from model.cal_flow import cal_flow
from utils.draw import *
from utils.load_data import *
from utils.make_file import make_output_dir
import numpy as np
import matplotlib
import time
import cv2
import argparse

matplotlib.use('TkAgg')
np.random.seed(0)

def get_track_para(d):
    d = d.astype(np.int32)
    track_id = int(d[-1])
    cx1, cy1, cx2, cy2 = d[0], d[1], d[2], d[3]
    cx = (cx1 + cx2) / 2
    cy = (cy1 + cy2) / 2
    return track_id, cx, cy

def init_sum(sumin, sumout, qin, qout):
    sumin = sumin - qin[frame % min_num]
    sumout = sumout - qout[frame % min_num]
    qin[frame % min_num] = 0
    qout[frame % min_num] = 0
    return sumin, sumout

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='buaa demo')
    parser.add_argument('--name', dest='name', type=str, default='demo_v1')
    parser.add_argument('--tr_type', dest='tracker_type', type=int, default=1)
    parser.add_argument('--if_detect', dest='if_detect', default=True)

    args = parser.parse_args()

    name = args.name
    tracker_type = args.tracker_type
    if_detect = args.if_detect

    if not os.path.exists('data/images/' + name) and if_detect:
        cutdata(name)

    print('==========================cut_data_done===============================')

    if not os.path.exists('data/' + name + '_result.txt') and if_detect:
        detect(name)

    print('==========================detect_done===============================')

    total_time = time.time()
    total_frames = 0
    ww, hh, line1, line2, line3, line4, line5, line6, line7, width, min_num = load_json(name)
    rotation, bias = get_parameters(line1, line2, line3, line4, line5, ww, hh, width)
    mark_list, vis, init_x, init_y, init_tim, pre_x, pre_y, pre_tim = make_zero(10000)

    qin = [0] * 100
    qout = [0] * 100
    sumin = 0
    sumout = 0

    if (tracker_type == 1):
        mot_tracker = DeepSort(model_path="model/deep_sort/deep/checkpoint/ckpt.t7")  # 实例化DEEPSORT tracker
    elif (tracker_type == 2):
        mot_tracker = Sort(max_age=3, min_hits=3, iou_threshold=0.3)  # 实例化 SORT tracker
    else:
        mot_tracker = Iou(max_age=3, min_hits=3, iou_threshold=0.3)  # 实例化 iou tracker

    seq_dets = np.loadtxt('data/' + name + '_result.txt', delimiter=',')  # 从txt文件中加载检测结果

    make_output_dir(name)

    with open('output/' + name + '/output.txt', 'w') as out_file:
        # 遍历每一帧
        for frame in range(int(seq_dets[:, 0].max())):
            frame += 1  # detection and frame numbers begin at 1

            sumin, sumout = init_sum(sumin, sumout, qin, qout)

            dets = seq_dets[seq_dets[:, 0] == frame, 1:6]  # 取出每一帧的检测结果 [x1,y1,w,h]

            total_frames += 1

            fn = os.path.join('data', 'images', name, '%06d.jpg' % frame)

            img_cv2 = cv2.imread(fn)

            if (tracker_type == 1):
                trackers = mot_tracker.update(dets[:, 0:4], dets[:, 4], img_cv2)
            else:
                trackers = mot_tracker.update(dets[:, 0:4])

            draw_boxes(img_cv2, trackers)

            for d in trackers:
                track_id, cx, cy = get_track_para(d)

                draw_average_v(cx, cy, rotation, bias, vis, track_id, frame,
                                         pre_x, pre_y, pre_tim, min_num, img_cv2)

                sumin, sumout = cal_flow(line6, line7, cx, cy, mark_list, track_id, qin, qout,
                                         init_x, init_y, init_tim, frame, min_num, sumin, sumout)

            draw_flow(sumin, sumout, line6, line7, img_cv2, frame, 0)

    total_time = time.time() - total_time
    if (tracker_type == 1):
        tracker_str = 'deep_sort'
    elif (tracker_type == 2):
        tracker_str = 'sort'
    else:
        tracker_str = 'iou'

    print("Demo name: %s, tracker type: %s" % (name, tracker_str))
    print("Total Tracking took: %.3f seconds for %d frames or %.1f FPS" % (
        total_time, total_frames, total_frames / total_time))

    print('==========================count_done===============================')