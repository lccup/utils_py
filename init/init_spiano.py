from utils_py.general import *
import utils_py as ut
import itertools
import shutil
import sys
import warnings
from typing import List, Dict
from abc import abstractmethod

import pickle
import cv2

from utils_py import spiano as sp
from utils_py.spiano import musicxml_to_png
from utils_py.spiano.cvplot import plt_imshow, plt
from utils_py.spiano.cvplot import (
    draw_boxes,
    draw_text,
    draw_points,
    draw_boxes_relations,
    draw_points_relations,
    move_boxes,
    move_point,
    points2boxes,
    format_location,
)

from utils_py.spiano.cvplot import COLORS, COLORS_CYCLE

# ------------------------------
# 路径
# ------------------------------

P_ROOT = str(Path("~/link/sp_gen_mxl").expanduser())
os.chdir(P_ROOT)
None if P_ROOT in sys.path else sys.path.append(P_ROOT)
P_ROOT = Path(P_ROOT)


def musicxml_to_png(
    xml_path,
    png_path=None,
    mscore_path=Path("/data/mscore/bin/mscore").expanduser(),
    options="",
):
    xml_path = Path(xml_path)
    if png_path is None:
        png_path = xml_path.with_suffix(".png")
    order = """
export PATH=$PATH:/data/mscore/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/data/mscore/qt6_libs
export QT_QPA_PLATFORM_PLUGIN_PATH=/data/mscore/plugins/platforms
export QT_QPA_PLATFORM=offscreen
"{}" -o "{}" "{}" {} > /dev/null 2>&1 
""".format(
        mscore_path, png_path, xml_path, options
    )
    os.system(order)


def rename_pdf_png(p_dir):
    """
    将pdf转换出的图片进行重命名
    """
    p_dir = Path(p_dir)
    assert p_dir.exists()
    fname_formatter = "{}_{{:02}}.png".format(p_dir.name)
    df = ut.df.iter_dir(p_dir)
    df = df[df["name"].str.match(".*_\\d+\\.png")]
    df["i"] = df["name"].str.extract("_(\\d+)\\.png").astype(int) + 1
    df = df.sort_values("i")
    df["name"] = df["i"].apply(lambda x: fname_formatter.format(x))
    for row in df.itertuples():
        row.path.replace(row.path.parent.joinpath(row.name))
