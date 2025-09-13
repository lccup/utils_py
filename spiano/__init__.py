import os
from pathlib import Path
import itertools

import numpy as np
import cv2

from utils_py.spiano import xml
from utils_py.spiano.cvplot import COLORS
from utils_py.spiano import cvplot as cvpl




def musicxml_to_png(xml_path,
                    png_path=None,
                    mscore_path=Path("/data1/musescore/musescore/mscore").expanduser(),
                    options=""
                    ):
    xml_path = Path(xml_path)
    # assert xml_path.exists(), "[not exists] {}".format(xml_path)
    if png_path is None:
        png_path = xml_path.with_suffix(".png")

    order = """ 
export LD_LIBRARY_PATH=/data1/musescore/musescore/libs:$LD_LIBRARY_PATH
export QT_QPA_PLATFORM_PLUGIN_PATH=/data1/musescore/musescore/plugins/platforms
export QT_QPA_PLATFORM=offscreen
"{}" -o "{}" "{}" {} > /dev/null 2>&1 
""".format(
        mscore_path, png_path, xml_path, options
    )

    # print(order)
    os.system(order)

def letterbox(img, dstSize=(3200, 3840)):
    ih, iw = img.shape[:2]
    w, h = dstSize
    color = list(map(int, img[0, 0]))
    if ih == h and iw == w:
        return img
    scale = min(w / iw, h / ih)
    nw = int(iw * scale)
    nh = int(ih * scale)

    dw = (w - nw) // 2
    dh = (h - nh) // 2

    scale_img = cv2.resize(img, [nw, nh])
    new_image = cv2.copyMakeBorder(
        scale_img, dh, h - dh - nh, dw, w - dw - nw, cv2.BORDER_CONSTANT, value=color)
    return new_image


def view_resutl(output_dir):
    from utils_py.df import np,iter_dir

    info = iter_dir(output_dir)
    info = info.reset_index(drop=True)
    mask_source = info["name"].str.contains("00_source")
    mask_mscore = info["name"].str.contains("00-1")
    assert mask_source.any()
    assert mask_mscore.any()
    img_source = cv2.imread(info.loc[np.argmax(mask_source),"path"])
    img_mscore = cv2.imread(info.loc[np.argmax(mask_mscore),"path"])

    img_mscore = letterbox(img_mscore)
    img_source = letterbox(img_source)

    return np.hstack([
        img_source,img_mscore
    ])


def x1x2y1y2_to_cxcywh(boxes):
    res = np.zeros(boxes.shape,dtype=boxes.dtype)
    res[:,0] = boxes[:,[0,2]].mean(axis=1)
    res[:,1] = boxes[:,[1,3]].mean(axis=1)
    res[:,2] = boxes[:,2]-boxes[:,0]
    res[:,3] = boxes[:,3]-boxes[:,1]
    return res

def cxcywh_to_x1x2y1y2(boxes):
    res = np.zeros(boxes.shape,dtype=boxes.dtype)
    res[:,0] = boxes[:,0] - boxes[:,2]/2
    res[:,1] = boxes[:,1] - boxes[:,3]/2
    res[:,2] = boxes[:,0] + boxes[:,2]/2
    res[:,3] = boxes[:,1] + boxes[:,3]/2
    return res