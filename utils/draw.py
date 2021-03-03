import cv2
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import os
from .geometry import getrealxy
from .geometry import getdis2

def cv2ImgAddText(img, text, left, top, textColor, textSize):
    cv2_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(cv2_img)
    # 创建一个可以在给定图像上绘图的对象
    draw = ImageDraw.Draw(pil_img)
    # 字体的格式
    font = ImageFont.truetype('font/test_font.ttf', textSize, encoding="utf-8")
    # 绘制文本
    draw.text((left, top), text, textColor, font=font)
    # 转换回OpenCV格式
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def compute_color_for_labels(label):
    """
    Simple function that adds fixed color depending on the class
    """
    palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)
    color = [int((p * (label ** 2 - label + 1)) % 255) for p in palette]
    return tuple(color)


def draw_boxes(img, trackers, offset=(0, 0)):
    if len(trackers) <= 0:
        return

    bbox = trackers[:, :4]
    identities = trackers[:, -1]

    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = [int(i) for i in box]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]
        # box text and bar
        id = int(identities[i]) if identities is not None else 0
        color = compute_color_for_labels(id)
        label = '{}{:d}'.format("", id)
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 2, 2)[0]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
        cv2.rectangle(img, (x1, y1), (x1 + t_size[0] + 3, y1 + t_size[1] + 4), color, -1)
        cv2.putText(img, label, (x1, y1 + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 2, [255, 255, 255], 2)


def draw_flow(in_volume, out_volume, line6, line7, img_cv2, frame, draw_line):
    text_in = 'down_flow: ' + str(in_volume) + ' cars/s'
    text_out = 'up_flow: ' + str(out_volume) + ' cars/s'
    text_tot = 'sum_flow: ' + str(in_volume + out_volume) + ' cars/s'

    cv2.putText(img_cv2, text_in, (200, 30), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
    cv2.putText(img_cv2, text_out, (200, 100), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
    cv2.putText(img_cv2, text_tot, (200, 170), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
    '''
    img_cv2 = cv2ImgAddText(img_cv2, text_in, 200, 30, (255, 0, 0), 25)
    img_cv2 = cv2ImgAddText(img_cv2, text_out, 200, 100, (255, 0, 0), 25)
    img_cv2 = cv2ImgAddText(img_cv2, text_tot, 200, 170, (255, 0, 0), 25)
    用这个方法能加中文，putText不行
    '''
    if draw_line:
        cv2.line(img_cv2, tuple(line6[0]), tuple(line6[1]), (0, 255, 0), 5)
        cv2.line(img_cv2, tuple(line7[0]), tuple(line7[1]), (0, 255, 0), 5)


def init_plot(plt):
    plt.ion()
    fig = plt.figure()
    ax1 = fig.add_subplot(111, aspect='equal')
    return fig, ax1

def draw_average_v(cx, cy, rotation, bias, vis, track_id, frame, pre_x, pre_y, pre_tim, min_num, img_cv2):
    if vis[track_id] == 1:
        real_v_a = getrealxy([cx, cy], rotation, bias)
        real_v_b = getrealxy([pre_x[track_id], pre_y[track_id]], rotation, bias)
        dis = getdis2(real_v_a, real_v_b)
        tim = (frame - pre_tim[track_id]) / min_num
        if tim != 0:
            text_velocity = str(round(dis / tim * 3.6, 2)) + "km/h"
            cv2.putText(img_cv2, text_velocity, (int(cx), int(cy)), cv2.FONT_HERSHEY_PLAIN, 1, [255, 255, 255], 2)

    if vis[track_id] == 0:
        vis[track_id] = 1
        pre_tim[track_id] = frame
        pre_x[track_id] = cx
        pre_y[track_id] = cy