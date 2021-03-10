import cv2
import os


def cutdata(name):
    # 打开视频
    video = cv2.VideoCapture('data/' + name + '.mp4')
    fps = video.get(cv2.CAP_PROP_FPS)
    frame_count = video.get(cv2.CAP_PROP_FRAME_COUNT)
    print('FPS: %d, Total frame: %d' % (fps, frame_count))

    save_path = 'data/images/' + name
    if os.path.exists(save_path) == 0:
        os.mkdir(save_path)

    import time
    temp = time.time()

    count = 1
    while video.isOpened():
        # 获取1帧
        check, frame = video.read()

        # 保存当前帧
        file_name = save_path + '/%06d.jpg' % count
        cv2.imwrite(file_name, frame)
        count += 1

        # 快进20ms，也就是50FPS
        pos_msec = int(video.get(cv2.CAP_PROP_POS_MSEC))
        # video.set(cv2.CAP_PROP_POS_MSEC, pos_msec + 20)

        # 视频长度1min=60000ms
        if pos_msec > 60000 or (not check) or cv2.waitKey(1) & 0xFF == ord('q'):
            break

    temp = time.time() - temp
    print("%.3f %.3f"%(temp, 1500/temp))
    video.release()
    cv2.destroyAllWindows()


