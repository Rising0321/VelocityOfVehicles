def cal_flow(line6, line7, cx, cy, mark_list, track_id, qin, qout, init_x, init_y, init_tim, frame, min_num, sumin, sumout):
    ty1 = (line6[1][1] - line6[0][1]) / (line6[1][0] - line6[0][0]) * (cx - 1) + line6[0][1]

    ty2 = (line7[1][1] - line7[0][1]) / (line7[1][0] - line7[0][0]) * (cx - 1) + line7[0][1]

    in_target_area = (cy >= ty1 and ty2 >= cy)  # (cx1 >= tx1 and cy1 >= ty1 and cx2 <= tx2 and cy2 <= ty2)

    out_target_area = not in_target_area

    if (out_target_area):
        if mark_list[track_id] == 0:
            mark_list[track_id] = 1

        elif mark_list[track_id] == 2 and abs(ty1 - cy) >= abs(ty2 - cy):
            '''
            real_v_a = getrealxy([cx, cy], rotation, bias)
            real_v_b = getrealxy([init_x[track_id], init_y[track_id]], rotation, bias)
            dis = getdis2(real_v_a, real_v_b)
            tim = (frame - init_tim[track_id]) / 30
            if tim != 0:
                text_velocity = dis / tim * 3.6
            print('%d %.2f %.2f %.2f %.2f' % (track_id, tim, dis, text_velocity, 18 / tim * 3.6),
                  file=out_file)
            '''
            mark_list[track_id] = 3
            qin[frame % min_num] += 1
            sumin += 1


        elif mark_list[track_id] == 4 and abs(ty1 - cy) < abs(ty2 - cy):
            '''
            real_v_a = getrealxy([cx, cy], rotation, bias)
            real_v_b = getrealxy([init_x[track_id], init_y[track_id]], rotation, bias)
            dis = getdis2(real_v_a, real_v_b)
            tim = (frame - init_tim[track_id]) / 30
            if tim != 0:
                text_velocity = dis / tim * 3.6
            print('%d %.2f %.2f %.2f %.2f' % (track_id, tim, dis, text_velocity, 18 / tim * 3.6),
                  file=out_file)
            '''
            mark_list[track_id] = 5
            qout[frame % min_num] += 1
            sumout += 1

    if (in_target_area):
        if mark_list[track_id] == 1:
            if abs(ty1 - cy) < abs(ty2 - cy):
                mark_list[track_id] = 2
                init_x[track_id] = cx
                init_y[track_id] = cy
                init_tim[track_id] = frame
            else:
                mark_list[track_id] = 4
                init_x[track_id] = cx
                init_y[track_id] = cy
                init_tim[track_id] = frame
    return sumin, sumout