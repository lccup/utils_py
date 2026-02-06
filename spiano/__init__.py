import os
from pathlib import Path
import math

import numpy as np
import cv2

from . import xml
from . import cvplot as cvpl
from .cvplot import COLORS


def x1x2y1y2_to_cxcywh(boxes):
    res = np.zeros(boxes.shape, dtype=boxes.dtype)
    res[:, 0] = boxes[:, [0, 2]].mean(axis=1)
    res[:, 1] = boxes[:, [1, 3]].mean(axis=1)
    res[:, 2] = boxes[:, 2] - boxes[:, 0]
    res[:, 3] = boxes[:, 3] - boxes[:, 1]
    return res


def cxcywh_to_x1x2y1y2(boxes):
    res = np.zeros(boxes.shape, dtype=boxes.dtype)
    res[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
    res[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
    res[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
    res[:, 3] = boxes[:, 1] + boxes[:, 3] / 2
    return res


def xywh_to_x1x2y1y2(boxes):
    res = np.zeros(boxes.shape, dtype=boxes.dtype)
    res[:, 0] = boxes[:, 0]
    res[:, 1] = boxes[:, 1]
    res[:, 2] = boxes[:, 0] + boxes[:, 2]
    res[:, 3] = boxes[:, 1] + boxes[:, 3]
    return res
