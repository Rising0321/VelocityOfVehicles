import math
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

def getdis2(p1, p2):
    return math.sqrt(((p1[0] - p2[0]) ** 2) + ((p1[1] - p2[1]) ** 2))