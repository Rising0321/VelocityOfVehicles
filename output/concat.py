import cv2

writer = cv2.VideoWriter('result_demov2.mp4', cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), 25, (1280, 720), True)

# **********设置帧的数量**********
total_frame = 1500
for frame_num in range(total_frame):
    frame_num += 1
    img_path = 'images/%06d.jpg' % frame_num
    read_img = cv2.imread(img_path)
    writer.write(read_img)
    print(frame_num)

writer.release()