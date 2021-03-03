import os
import json

with open(f"utils/config.json") as file:
    config = json.load(file)


def load_attr(name, attr):
    return config[name][attr]


def load_json(name):
    ww = load_attr(name, "ww")
    hh = load_attr(name, "hh")
    line1 = load_attr(name, "line1")#[0]
    line2 = load_attr(name, "line2")#[0]
    line3 = load_attr(name, "line3")#[0]
    line4 = load_attr(name, "line4")#[0]
    line5 = load_attr(name, "line5")#[0]
    line6 = load_attr(name, "line6")#[0]
    line7 = load_attr(name, "line7")#[0]
    width = load_attr(name, "width")
    frame = load_attr(name, "frame")
    return ww, hh, line1, line2, line3, line4, line5, line6, line7, width, frame

def make_zero(num):
    return [0]*num, [0]*num, [0]*num, [0]*num, [0]*num, [0]*num, [0]*num, [0]*num
