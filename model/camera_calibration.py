import math

import torch


def line(p):
    p1, p2 = p
    A = (p1[1] - p2[1])
    B = (p2[0] - p1[0])
    C = (p1[0] * p2[1] - p2[0] * p1[1])
    return A, B, -C


def intersection(L1, L2):
    D = L1[0] * L2[1] - L1[1] * L2[0]
    Dx = L1[2] * L2[1] - L1[1] * L2[2]
    Dy = L1[0] * L2[2] - L1[2] * L2[0]
    if D != 0:
        x = Dx / D
        y = Dy / D
        return x, y


def getFootPoint(point, line_p1, line_p2):
    """
    求点到直线的的垂足
    :param point:
    :param line_p1:
    :param line_p2:
    :return:
    """
    x0 = point[0]
    y0 = point[1]

    x1 = line_p1[0]
    y1 = line_p1[1]

    x2 = line_p2[0]
    y2 = line_p2[1]

    k = -((x1 - x0) * (x2 - x1) + (y1 - y0) * (y2 - y1)) / ((x2 - x1) ** 2 + (y2 - y1) ** 2) * 1.0

    xn = k * (x2 - x1) + x1
    yn = k * (y2 - y1) + y1

    return xn, yn


def getdis2(p1, p2):
    return math.sqrt(((p1[0] - p2[0]) ** 2) + ((p1[1] - p2[1]) ** 2))


def getdis3(p1, p2):
    return math.sqrt(((p1[0] - p2[0]) ** 2) + ((p1[1] - p2[1]) ** 2) + ((p1[2] - p2[2]) ** 2))


def cross(x, y):
    """
    求x和y的叉积
    :param x:
    :param y:
    :return:
    """
    return x[1] * y[2] - x[2] * y[1], x[2] * y[0] - x[0] * y[2], x[0] * y[1] - x[1] * y[0]


def getmul(p1, p2):
    tempp1 = torch.tensor(p1)
    tempp2 = torch.tensor(p2)
    p3 = tempp1.mm(tempp2.float().transpose(0, 1))
    return p3.transpose(0, 1).tolist()


def getsec(p1_, p2_, o, p2):
    vec1 = [p2_[0] - p1_[0], p2_[1] - p1_[1], p2_[2] - p1_[2]]
    vec2 = [p2[0] - o[0], p2[1] - o[1], p2[2] - o[2]]
    t = (vec1[1] * p1_[0] - vec1[0] * p1_[1]) / (vec1[1] * vec2[0] - vec1[0] * vec2[1])
    return vec2[0] * t, vec2[1] * t, vec2[2] * t


def get_parameters(line1, line2, line3, line4, line5, w, h, width):
    """
    :param line1: 车道线1
    :param line2: 车道线2
    :param line3: 两个车道线的连线1
    :param line4: 两个车道线的连线2
    :param w: 画幅长
    :param h: 画幅宽
    :return: 矩阵K，R，T
    """
    # 求车道线1和车道线2的交点
    L1 = line(line1)
    L2 = line(line2)
    v1 = intersection(L1, L2)  # 消失点1
    # 求连线1和连线2的交点
    L3 = line(line3)
    L4 = line(line4)
    v2 = intersection(L3, L4)  # 消失点2

    # print(v1,v2)

    oi = [w / 2, h / 2]  # 画幅中心
    v1 = [v1[0] - oi[0], v1[1] - oi[1]]
    v2 = [v2[0] - oi[0], v2[1] - oi[1]]

    vi = getFootPoint(oi, v1, v2)  # 求vi

    # print(v1, v2, vi)

    ocvi = math.sqrt(getdis2(v1, vi) * getdis2(vi, v2))

    f = math.sqrt((ocvi ** 2) - (getdis2(oi, vi) ** 2))  # 求ocvi的长度，使用公式4

    K = [[f, 0, oi[0]], [0, f, oi[1]], [0, 0, 1]]  # 得到矩阵K

    # print(f)

    len_ocv1 = getdis3([v1[0], v1[1], f], [0, 0, 0])  # Xc = ||ocvi||
    len_ocv2 = getdis3([v2[0], v2[1], f], [0, 0, 0])  # Yc = ||ocv2||

    y = [v1[0] / len_ocv1, v1[1] / len_ocv1, f / len_ocv1]
    x = [v2[0] / len_ocv2, v2[1] / len_ocv2, f / len_ocv2]
    z = cross(x, y)  # 求x和y的叉积
    R = [[x[0], y[0], z[0]], [x[1], y[1], z[1]], [x[2], y[2], z[2]]]

    # print(getdis3(z,[0,0,0]))

    # print(R)

    p1 = [0, 0, 0]
    p2 = [width, 0, 0]
    p1_m = getmul(R, [p1])  # 旋转后的p1
    p1_m = p1_m[0]
    p2_m = getmul(R, [p2])  # 旋转后的p2
    p2_m = p2_m[0]

    # print(p1_m,p2_m)

    p1px = line5[0]
    p2px = line5[1]

    i1m = [p1px[0] - oi[0], p1px[1] - oi[1]]  # 公式8

    p1_ = [i1m[0], i1m[1], f]
    p2_ = [i1m[0] + p2_m[0] - p1_m[0], i1m[1] + p2_m[1] - p1_m[1], f + p2_m[2] - p1_m[2]]  # 公式9
    q = getsec(p1_, p2_, [0, 0, 0], [p2px[0] - oi[0], p2px[1] - oi[1], f])

    # print("fuck")
    # print(q)

    len_ocp1_ = getdis3(p1_, [0, 0, 0])
    len_p1p2 = getdis3(p2, p1)
    #print(len_p1p2)
    len_p1_q = getdis3(p1_, q)
    d = len_ocp1_ * len_p1p2 / len_p1_q  # 公式11

    alpha = d / len_ocp1_

    # print(alpha/100)

    T = [[p1_[0] * alpha], [p1_[1] * alpha], [p1_[2] * alpha]]
    # print(T)
    rotation = torch.tensor(K).mm(torch.tensor(R))
    rotation = -rotation
    bias = torch.tensor(K).mm(torch.tensor(T))

    if getrealxy(line5[1], rotation, bias)[0] < 0:
        rotation = -rotation
    return rotation, bias


def getplane(node, rotation, bias):
    node_rotated = rotation.float().mm(node.float().transpose(0, 1))
    node_biased = node_rotated + bias
    # print(node_rotated,bias,node_biased)
    # print(bias.shape)
    return node_biased[0][0] / node_biased[2][0], node_biased[1][0] / node_biased[2][0]

def calibration(world_node):

    line1 = [594, 679], [580, 581]
    line2 = [498, 332], [472, 654]
    line3 = [267, 471], [317, 472]
    line4 = [239, 519], [294, 523]
    rotation, bias = get_parameters(line1, line2, line3, line4)
    plane_node = getplane(torch.tensor([world_node]), rotation, bias)
    return plane_node

def getrealxy(plane_node, rotation, bias):
    u = plane_node[0]
    v = plane_node[1]

    a1 = rotation[0][0] - rotation[2][0] * u
    a2 = rotation[1][0] - rotation[2][0] * v

    b1 = rotation[0][1] - rotation[2][1] * u
    b2 = rotation[1][1] - rotation[2][1] * v

    c1 = bias[2] * u - bias[0]
    c2 = bias[2] * v - bias[1]

    y = (c1 * a2 - c2 * a1) / (b1 * a2 - b2 * a1)
    x = (b1 * c2 - b2 * c1) / (b1 * a2 - b2 * a1)

    return [x, y]