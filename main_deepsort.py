from __future__ import print_function

import sys
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from model.camera_calibration import get_parameters
from model.camera_calibration import getdis2
from model.cut_data import cutdata
from model.detect import detect
from model.deep_sort.deep_sort import DeepSort
from utils.draw import cv2ImgAddText
from utils.geometry import getrealxy
from utils.draw import draw_boxes
import numpy as np
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from skimage import io

import time
import argparse

import cv2

np.random.seed(0)


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='SORT demo')
    # 是否实时演示结果
    parser.add_argument('--display', dest='display', help='Display online tracker output (slow) [False]',
                        action='store_true')
    # 是否保存每一帧到文件
    parser.add_argument('--save-frame', dest='save_frame',
                        help='Save frames with trackers\' bbox to image file [False]',
                        action='store_true', default=True)
    # 存放检测结果的路径，默认是data
    parser.add_argument("--seq_path", help="Path to detections.", type=str, default='data')
    # 阶段，默认是train
    parser.add_argument("--phase", help="Subdirectory in seq_path.", type=str, default='train')
    # N_lost，默认是1
    parser.add_argument("--max_age",
                        help="Maximum number of frames to keep alive a track without associated detections.",
                        type=int, default=3)
    # 未确认到确认所需的帧数，默认是3
    parser.add_argument("--min_hits",
                        help="Minimum number of associated detections before track is initialised.",
                        type=int, default=3)
    # IoU阈值，默认是0.3
    parser.add_argument("--iou_threshold", help="Minimum IOU for match.", type=float, default=0.3)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    name = 'buaa1_2'
    if not os.path.exists('data/images/' + name):
        cutdata(name)

    print('==========================cut_data_done===============================')

    if not os.path.exists('data/' + name + '_result.txt'):
        detect(name)

    print('==========================detect_done===============================')

    # all train
    args = parse_args()
    display = args.display
    save_frame = args.save_frame
    phase = args.phase
    total_time = 0.0
    total_frames = 0

    '''
    # demo_v1
    ww = 1280
    hh = 720
    line1 = [594, 679], [580, 581]
    line2 = [498, 332], [472, 654]
    line3 = [267, 471], [317, 472]
    line4 = [239, 519], [294, 523]
    line5 = line3 #[639, 440], [655, 488]
    width = 3.25
    tx1, ty1, tx2, ty2 = 1, 420, 1279, 719
    '''

    '''
    # demo_v2
    ww = 854
    hh = 480
    line1 = [290, 354], [174, 405]
    line2 = [327, 198], [274, 212]
    line3 = [529, 339], [435, 292]
    line4 = [328, 198], [495, 265]
    line5 = line3
    width = 3.25
    tx1, ty1, tx2, ty2 = 1, 206 - 100, 853, 479 - 100
    '''

    # buaa1_2
    ww = 1920
    hh = 1080
    line1 = [318, 978], [1584, 944]
    line2 = [632, 656], [1245, 639]
    line3 = [719, 753], [827, 549]
    line4 = [1110, 659], [1012, 482]
    line5 = [566, 1056], [648, 892]
    width = 4.0
    tx1, ty1, tx2, ty2 = 1, 206 - 100, 853, 479 - 100


    rotation, bias = get_parameters(line1, line2, line3, line4, line5, ww, hh, width)
    if getrealxy(line5[1], rotation, bias)[0] < 0:
        rotation = -rotation
    # 目标区域

    # 标记列表 0 不存在, -1 在外面, 1 在里面
    mark_list = [0] * 10000
    init_x = [0] * 10000
    init_y = [0] * 10000
    pre_x = [0] * 10000
    pre_y = [0] * 10000
    pre_tim = [0] * 10000
    vis = [0] * 10000
    init_tim = [0] * 10000

    # print(mark_list)
    # 生成32个随机颜色
    colours = np.random.rand(32, 3)  # used only for display
    # print('colors:', colours)
    if display:
        # 如果要展示必须有原始图片
        # if not os.path.exists('mot_benchmark'):
        #     print('\n\tERROR: mot_benchmark link not found!\n\n    Create a symbolic link to the MOT benchmark\n    (https://motchallenge.net/data/2D_MOT_2015/#download). E.g.:\n\n    $ ln -s /path/to/MOT2015_challenge/2DMOT2015 mot_benchmark\n\n')
        #     exit()
        plt.ion()
        fig = plt.figure()
        ax1 = fig.add_subplot(111, aspect='equal')
    # 输出结果目录
    if not os.path.exists('output'):
        os.makedirs('output')
    # 匹配检测结果文件det.txt
    # pattern = os.path.join(args.seq_path, phase, '*', 'det', 'det.txt')
    # 遍历每个任务
    # for seq_dets_fn in glob.glob(pattern):
    # 实例化SORT tracker
    mot_tracker = DeepSort(model_path="model/deep_sort/deep/checkpoint/ckpt.t7"
                           )  # create instance of the SORT tracker
    # 从txt文件中加载检测结果
    seq_dets = np.loadtxt('data/' + name + '_result.txt', delimiter=',')
    # 任务名称
    # seq = seq_dets_fn[pattern.find('*'):].split(os.path.sep)[0]
    # 打开输出结果文件

    if not os.path.exists('output/' + name):
        os.mkdir('output/' + name)

    with open('output/' + name + '/output.txt', 'w') as out_file:
        # print("Processing %s." % (seq))
        in_volume, out_volume = 0, 0
        prev_in_volume, prev_out_volume = 0, 0
        # 遍历每一帧
        for frame in range(int(seq_dets[:, 0].max())):
            frame += 1  # detection and frame numbers begin at 1
            # 取出每一帧的检测结果 [x1,y1,w,h]
            dets = seq_dets[seq_dets[:, 0] == frame, 1:6]
            # dets[:, 2:4] += dets[:, 0:2]  # convert [x1,y1,w,h] to [x1,y1,x2,y2]
            total_frames += 1

            if display:
                fn = os.path.join('data', 'images', name, '%06d.jpg' % frame)
                im = io.imread(fn)
                ax1.imshow(im)
                plt.title('Tracked Targets')

            fn = os.path.join('data', 'images', name, '%06d.jpg' % frame)
            img_cv2 = cv2.imread(fn)

            start_time = time.time()
            trackers = mot_tracker.update(dets[:, 0:4], dets[:, 4], img_cv2)
            cycle_time = time.time() - start_time
            total_time += cycle_time

            volume_need_update = (frame - 1) % 25 == 0

            if len(trackers) > 0:
                bbox_tlwh = []
                bbox_xyxy = trackers[:, :4]
                identities = trackers[:, -1]
                img_cv2 = draw_boxes(img_cv2, bbox_xyxy, identities)

            for d in trackers:
                # frame,ID,x1,y1,w,h,conf,-1,-1,-1
                #print('%d,%d,%.2f,%.2f,%.2f,%.2f,1,-1,-1,-1' % (frame, d[-1], d[0], d[1], d[2] - d[0], d[3] - d[1]),
                #      file=out_file)
                d = d.astype(np.int32)

                track_id = int(d[-1])

                cx1, cy1, cx2, cy2 = d[0], d[1], d[2], d[3]
                # print(f'cx1, cy1, cx2, cy2: {cx1}, {cy1}, {cx2}, {cy2}')
                # print(f'tx1, ty1, tx2, ty2: {tx1}, {ty1}, {tx2}, {ty2}')

                cx = (cx1 + cx2) / 2
                cy = (cy1 + cy2) / 2

                # 显示实时速度非常不准

                if vis[track_id] == 1:
                    real_v_a = getrealxy([cx, cy], rotation, bias)
                    real_v_b = getrealxy([pre_x[track_id], pre_y[track_id]], rotation, bias)
                    dis = getdis2(real_v_a, real_v_b)
                    tim = (frame - pre_tim[track_id]) / 50
                    if tim != 0 :
                        text_velocity = str(round(dis / tim * 3.6, 2)) + "km/h"
                        img_cv2 = cv2ImgAddText(img_cv2, text_velocity, cx, cy, (255, 255, 255), 15)

                if vis[track_id] == 0:
                    vis[track_id] = 1
                    pre_tim[track_id] = frame
                    pre_x[track_id] = cx
                    pre_y[track_id] = cy

                in_target_area = (cy >= ty1 and ty2 >= cy)  # (cx1 >= tx1 and cy1 >= ty1 and cx2 <= tx2 and cy2 <= ty2)

                if (in_target_area):
                    if mark_list[track_id] == 0:
                        if abs(ty1 - cy) < abs(ty2 - cy):
                            mark_list[track_id] = 1
                            init_x[track_id] = cx
                            init_y[track_id] = cy
                            init_tim[track_id] = frame
                        else:
                            mark_list[track_id] = 3
                            init_x[track_id] = cx
                            init_y[track_id] = cy
                            init_tim[track_id] = frame

                    elif mark_list[track_id] == 1:
                        if abs(ty1 - cy) >= abs(ty2 - cy):
                            mark_list[track_id] = 2
                            in_volume += 1
                    elif mark_list[track_id] == 3:
                        if abs(ty1 - cy) < abs(ty2 - cy):
                            mark_list[track_id] = 4
                            out_volume += 1
                if display:
                    d = d.astype(np.int32)
                    ax1.add_patch(patches.Rectangle((d[1], d[2]), d[3] - d[1], d[4] - d[2], fill=False, lw=3,
                                                    ec=colours[d[0] % 32, :]))

            if display:
                fig.canvas.flush_events()
                plt.draw()
                ax1.cla()

            if save_frame:
                if volume_need_update:
                    prev_in_volume = in_volume
                    prev_out_volume = out_volume
                    in_volume = 0
                    out_volume = 0

                '''
                text_in = '下行流量: ' + str(prev_in_volume) + ' 辆/秒'
                text_out = '上行流量: ' + str(prev_out_volume) + ' 辆/秒'
                text_tot = '总流量: ' + str(prev_in_volume + prev_out_volume) + ' 辆/秒'
                img_cv2 = cv2ImgAddText(img_cv2, text_in, 200, 30, (255, 0, 0), 25)
                img_cv2 = cv2ImgAddText(img_cv2, text_out, 200, 100, (255, 0, 0), 25)
                img_cv2 = cv2ImgAddText(img_cv2, text_tot, 200, 170, (255, 0, 0), 25)
                '''

                cv2.line(img_cv2, (tx1, ty1), (tx2, ty1), (0, 255, 0), 5)
                cv2.line(img_cv2, (tx1, ty2), (tx2, ty2), (0, 255, 0), 5)
                cv2.line(img_cv2, (tx1, (ty2 + ty1) // 2), (tx2, (ty2 + ty1) // 2), (255, 0, 0), 5)
                fn = os.path.join('output', 'images', '%06d.jpg' % frame)
                print(fn)
                cv2.imwrite(fn, img_cv2)

    print("Total Tracking took: %.3f seconds for %d frames or %.1f FPS" % (
        total_time, total_frames, total_frames / total_time))

    if display:
        print("Note: to get real runtime results run without the option: --display")

    print('==========================count_done===============================')
