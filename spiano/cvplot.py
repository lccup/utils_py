import itertools

import numpy as np
import cv2
import matplotlib.pyplot as plt


def hex_to_bgr(hex_color):
    """
    将十六进制颜色代码转换为 RGB 元组。

    Parameters
    ----------
        hex_color (str): 十六进制颜色字符串（如 "#FF0000" 或 "FF0000"）。

    Returns
    ----------
        tuple: (R, G, B)，每个分量范围 0-255。

    Examples
    --------
        >>> hex_to_rgb("#FF0000")
        (255, 0, 0)
    """
    # 去除可能的 '#' 符号
    hex_color = hex_color.lstrip('#')

    # 检查长度是否合法（支持 3 或 6 位十六进制）
    if len(hex_color) == 3:
        hex_color = ''.join([c * 2 for c in hex_color])  # 扩展 "F00" → "FF0000"
    elif len(hex_color) != 6:
        raise ValueError("十六进制颜色代码必须是 3 或 6 位字符（如 '#FF0000' 或 'F00'）")

    # 转换为 RGB 整数
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return b, g, r


COLORS = {
    "blue": [hex_to_bgr(i)
             for i in "#CEE1FF,#A3C8FD,#78AEFE,#4C94FC,#237AFF,#0363EA,#0A50B8,#073D8C,#04295F,#021533".split(",")],
    "indigo": [hex_to_bgr(i)
               for i in "#DFCEFC,#C5A4FA,#A97AF7,#8F50F4,#7324F7,#5C08E0,#4B0DB0,#380886,#26065B,#130230".split(",")],
    "purple": [hex_to_bgr(i)
               for i in "#E1D8F2,#C8B7E7,#AF96DC,#9575D0,#7B51C8,#643AB1,#51318C,#3D246B,#291848,#150C26".split(",")],
    "pink": [hex_to_bgr(i)
             for i in "#F7D6E5,#EFB1D0,#E88DBA,#DF69A5,#DC438F,#C52B78,#9B2660,#761C49,#511232,#2B0A19".split(",")],
    "red": [hex_to_bgr(i)
            for i in "#F8D7D9,#F1B2B9,#EB8E97,#E46A76,#E24553,#CB2D3C,#A02732,#7A1D25,#53131A,#2C0B0D".split(",")],
    "orange": [hex_to_bgr(i)
               for i in "#FFE5D0,#FDCDA6,#FEB77C,#FCA052,#FF8927,#EA720C,#B85C0F,#8C460A,#5F2F07,#331904".split(",")],
    "yellow": [hex_to_bgr(i)
               for i in "#FFF2CC,#FEE7A1,#FFDC75,#FDD049,#FFC81D,#ECB100,#B98C06,#8D6B03,#604803,#332600".split(",")],
    "green": [hex_to_bgr(i)
              for i in "#D1E6DD,#A8D1BE,#7FBC9F,#56A681,#2C9161,#117B4B,#13623D,#0D4B2E,#093220,#051A11".split(",")],
    "teal": [hex_to_bgr(i)
             for i in "#D2F3EA,#AAEAD7,#82E1C5,#5BD7B2,#33D0A1,#18B98A,#18936E,#117054,#0C4B38,#06271E".split(",")],
    "cyan": [hex_to_bgr(i)
             for i in "#CEF3FC,#A3EBF9,#78E1F6,#4DD7F2,#22D1F5,#04BADE,#0A93AF,#077085,#044C5A,#022730".split(",")],
    "gray": [hex_to_bgr(i)
             for i in "#F8F8F9,#EAEDF0,#E0E4E8,#D4D9DE,#C0C7CE,#9098A1,#5C656C,#434A51,#31373D,#202528".split(",")],

}


class ColorCycle:
    def __init__(self):
        self.base_colors = np.array([
                COLORS["red"][1:6:2],
                COLORS["blue"][1:6:2],
                COLORS["orange"][1:6:2],
                COLORS["green"][1:6:2],
                COLORS["cyan"][1:6:2],
            ]).transpose((1, 0, 2)).reshape(-1, 3).tolist()
        # itertools.cycle(
        # itertools.chain.from_iterable(zip())
        # )
        self.index = 0

    def __iter__(self):
        return self

    def __next__(self):
        color = self.base_colors[self.index]
        self.index = (self.index + 1) % len(self.base_colors)
        return color

    def reset(self):
        self.index = 0

COLORS_CYCLE = ColorCycle()


def format_location(locations, formatter="{row_id:.0f},{measure_id:.0f},{staff:.0f}"):
    return np.apply_along_axis(
        lambda arr: formatter.format(
            row_id=arr[0],
            measure_id=arr[1],
            staff=arr[2],
        ), axis=1, arr=locations
    )


def get_note_mask(notes, row_id=None, measure_id=None, staff=None):
    note_mask = np.full(notes.shape[0], True, dtype=np.bool_)
    if row_id:
        note_mask = note_mask & (notes[:, 5] == row_id)
    if measure_id:
        note_mask = note_mask & (notes[:, 6] == measure_id)
    if staff:
        note_mask = note_mask & (notes[:, 7] == staff)
    return note_mask


def cal_center_point(boxes):
    return np.hstack([
        boxes[:, [0, 2]].mean(axis=1)[:, np.newaxis],
        boxes[:, [1, 3]].mean(axis=1)[:, np.newaxis]
    ])


def move_point(points, offset_x=0, offset_y=0):
    return np.hstack([
        points[:, 0][:, np.newaxis] + offset_x,
        points[:, 1][:, np.newaxis] + offset_y,
    ])


def move_boxes(boxes, offset_x=0, offset_y=0):

    boxes[:, [0, 2]] += offset_x
    boxes[:, [1, 3]] += offset_y
    return boxes


def draw_boxes(img, boxes, color=COLORS["red"][2], thickness=3):
    def handel(box):
        cv2.rectangle(img, box[:2], box[2:], color=color, thickness=thickness)
    boxes = boxes.astype(int)
    np.apply_along_axis(
        lambda box: handel(box),
        axis=1, arr=boxes
    )
    return img


def draw_points(img, points, radius=15, color=COLORS["blue"][3], thickness=-1):
    for point in points.astype(int):
        cv2.circle(img, point, radius, color, thickness)
    return img


def draw_text(img, points, texts, fontscale=1.5, color=COLORS["pink"][3], thickness=3):
    assert points.shape[0] == len(texts)
    points = points.astype(int)
    for i in range(points.shape[0]):
        cv2.putText(img, texts[i],
                    points[i, :2],
                    cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=fontscale,
                    color=color, thickness=thickness
                    )
    return img


def draw_boxes_relations(img, relations, boxes1, boxes2,
                         color_box=COLORS["red"][3],
                         color_line=COLORS["blue"][3],
                         thickness=3):
    relations = relations.astype(int)
    boxes1 = boxes1.astype(int)
    boxes2 = boxes2.astype(int)
    cp1 = cal_center_point(boxes1).astype(int)
    cp2 = cal_center_point(boxes2).astype(int)

    def handel(arr):
        if arr[0] < 0 or arr[1] < 0:
            return
        cv2.rectangle(img, boxes1[arr[0], 2:], boxes1[arr[0],
                      :2], color=color_box, thickness=thickness)
        cv2.rectangle(img, boxes2[arr[1], 2:], boxes2[arr[1],
                      :2], color=color_box, thickness=thickness)
        cv2.line(img, cp1[arr[0]], cp2[arr[1]],
                 color=color_line, thickness=thickness)
    np.apply_along_axis(handel, axis=1, arr=relations)
    return img


def draw_points_relations(img, points1, points2, radius=15,
                          color_point1=COLORS["cyan"][5],
                          color_point2=COLORS["blue"][5],
                          color_line=COLORS["red"][5],
                          thickness=- 1):
    def handle(img, arr, radius, color_point1, color_point2, color_line, thickness):
        cv2.circle(img, arr[:2], radius, color_point1, thickness)
        cv2.circle(img, arr[2:], radius, color_point2, thickness)
        cv2.line(img, arr[:2], arr[2:], color_line, thickness=5)

    points = np.hstack([
        points1.astype(int),
        points2.astype(int),
    ])
    np.apply_along_axis(
        lambda arr: handle(img, arr, radius, color_point1, color_point2,
                           color_line, thickness),
        axis=1, arr=points
    )
    return img


def points2boxes(points1, points2):
    points = np.hstack([
        points1.astype(int),
        points2.astype(int),
    ])

    boxes = np.vstack([
        points[:, [0, 2]].min(axis=1),
        points[:, [1, 3]].min(axis=1),
        points[:, [0, 2]].max(axis=1),
        points[:, [1, 3]].max(axis=1),
    ]).T

    boxes[:, 3] = np.where(
        boxes[:, 3] - boxes[:, 1] < 20,
        boxes[:, 1] + 20, boxes[:, 3]

    )

    return boxes


def plt_imshow(img, rc=None):
    plt.close("all")
    rc_default = {
        "axes.edgecolor": "white",
        "figure.dpi": 220,
        "xtick.labelbottom": False,
        "xtick.bottom": False,
        "ytick.labelleft": False,
        "ytick.left": False,
        # "axes.xmargin": .3,
        # "axes.ymargin": .3,
    }
    if rc:
        rc_default.update(rc)
    with plt.rc_context(rc=rc_default):
        if len(img.shape) == 3:
            img = img[..., ::-1]
        plt.imshow(img)
    plt.show()
