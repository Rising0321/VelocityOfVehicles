from __future__ import print_function

import sys
import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
from model.camera_calibration import get_parameters
from model.detect_updated import *
from model.deep_sort.deep_sort import DeepSort
from model.cal_flow import cal_flow
from utils.draw import *
from utils.load_data import *
from utils.make_file import make_output_dir
import numpy as np
import matplotlib
import time
import cv2
from model.tracker import Sort

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

    name = 'demo_v1'
    total_frames = 0
    detect_time = 0.0
    track_time = 0.0
    draw_time = 0.0
    ww, hh, line1, line2, line3, line4, line5, line6, line7, width, min_num = load_json(name)
    rotation, bias = get_parameters(line1, line2, line3, line4, line5, ww, hh, width)
    mark_list, vis, init_x, init_y, init_tim, pre_x, pre_y, pre_tim = make_zero(10000)

    qin = [0] * 100
    qout = [0] * 100
    sumin = 0
    sumout = 0

    detect_model = init_detect_model()

    #mot_tracker = DeepSort(model_path="model/deep_sort/deep/checkpoint/ckpt.t7")  # 实例化SORT tracker
    mot_tracker = Sort(max_age=3, min_hits=3, iou_threshold=0.3)

    make_output_dir(name)

    video = cv2.VideoCapture('data/' + name + '.mp4')

    frame = 0

    writer = cv2.VideoWriter("result_with_sort_" + name + ".mp4", cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), min_num, (ww, hh), True)

    total_time = time.time()
    while video.isOpened():

        checked, img = video.read()

        if not checked:
            break

        frame += 1

        if frame > 100 :
            break

        print(frame)

        sumin, sumout = init_sum(sumin, sumout, qin, qout)

        temp = time.time()

        dets = detect_oneframe(detect_model, img)  # 取出每一帧的检测结果 [x1,y1,w,h]

        detect_time = detect_time + time.time() - temp

        dets = np.array(dets)

        temp = time.time()

        total_frames += 1

        trackers = mot_tracker.update(dets)

        track_time = track_time + time.time() - temp

        draw_boxes(img, trackers)

        temp = time.time()

        for d in trackers:
            track_id, cx, cy = get_track_para(d)

            draw_average_v(cx, cy, rotation, bias, vis, track_id, frame,
                                     pre_x, pre_y, pre_tim, min_num, img)

            sumin, sumout = cal_flow(line6, line7, cx, cy, mark_list, track_id, qin, qout,
                                     init_x, init_y, init_tim, frame, min_num, sumin, sumout)

        draw_flow(sumin, sumout, line6, line7, img, frame, 0)

        writer.write(img)

        draw_time = draw_time + time.time() - temp

    video.release()
    cv2.destroyAllWindows()
    writer.release()

    total_time = time.time() - total_time
    print("Total Tracking took: %.3f %.3f %.3f %.3f seconds for %d frames or %.1f FPS" % (
        total_time, detect_time, track_time, draw_time, total_frames, total_frames / total_time))

