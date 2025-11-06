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
from utils_py.spiano import letterbox, musicxml_to_png, view_resutl
from utils_py.spiano.cvplot import plt_imshow, plt
from utils_py.spiano.cvplot import draw_boxes, draw_text, draw_points
from utils_py.spiano.cvplot import draw_boxes_relations, draw_points_relations
from utils_py.spiano.cvplot import move_boxes, move_point, points2boxes, format_location

from utils_py.spiano.cvplot import COLORS, COLORS_CYCLE

DICT_SLUR_COLOR = {
    "point_start": COLORS["red"][1],
    "point_stop": COLORS["cyan"][1],
    "line": COLORS["green"][3],
    "stem_start": COLORS["red"][3],
    "stem_stop": COLORS["cyan"][3],
    "note_start": COLORS["red"][5],
    "note_stop": COLORS["cyan"][5],
}

# ------------------------------
# 路径
# ------------------------------

P_ROOT = str(Path("~/link/piano-ai").expanduser())
os.chdir(P_ROOT)
None if P_ROOT in sys.path else sys.path.append(P_ROOT)
P_ROOT = Path(P_ROOT)


P_UNIT_TEST = P_ROOT.joinpath("share.pyc/db_unit_test")
P_DB_IMAGE = P_ROOT.joinpath("db_image.pyc")

MAP_PATH = {
    "db_image": P_DB_IMAGE,
    "db_image_images": P_DB_IMAGE.joinpath("images"),
    "db_image_badcase": Path(""),
    "db_image_badcase_complex": P_DB_IMAGE.joinpath(
        "complex_music_images_4090_badcase"
    ),
    "db_image_pos": P_DB_IMAGE.joinpath("PositiveSampleImages"),
    "db_image_pdf_imslp": Path(""),
    "workspace": Path(""),
    "artificial": P_DB_IMAGE.joinpath("artificial_muxicxml"),
}


TEST_SET = {
    "虚实线": [
        MAP_PATH["artificial"].joinpath("line_group_qinghuaci-1.png"),
        MAP_PATH["artificial"].joinpath("lined_strength_qinghuaci_1-1.png"),
        MAP_PATH["artificial"].joinpath("pedal_qinghuaci_1-1.png"),
        MAP_PATH["artificial"].joinpath("lined_speed_qinghuaci_1-1.png"),
        MAP_PATH["artificial"].joinpath("lined_strength_debug_1.png"),
        MAP_PATH["artificial"].joinpath("line_soild_qinghuaci_1.png"),
        MAP_PATH["artificial"].joinpath("qinghuaci-1.png"),
    ],
    "高低八度": [
        MAP_PATH["db_image_images"].joinpath(
            "Liszt_-_Ballade_No._1_in_Db_Major_S._170/Liszt_-_Ballade_No._1_in_Db_Major_S._170_3-3.png"
        ),
        MAP_PATH["db_image_images"].joinpath(
            "Liszt_-_Ballade_No._1_in_Db_Major_S._170/Liszt_-_Ballade_No._1_in_Db_Major_S._170_11-11.png"
        ),
        MAP_PATH["db_image_images"].joinpath(
            "Liszt_-_Ballade_No._1_in_Db_Major_S._170/Liszt_-_Ballade_No._1_in_Db_Major_S._170_13-13.png"
        ),
        MAP_PATH["db_image_images"].joinpath(
            "Scherzo_No.1_Op.39/Scherzo_No.1_Op.39_3-3.png"
        ),
        MAP_PATH["db_image_images"].joinpath(
            "Turkish_March_-_Volodos/Turkish_March_-_Volodos-9.png"
        ),
        # MAP_PATH["db_image_badcase"].joinpath(
        #     "694071617332031488_0.jpg"),
    ],
    "跳房子": [
        MAP_PATH["db_image"].joinpath("line_group_qinghuaci-1.png"),
        MAP_PATH["db_image_images"].joinpath(
            "Suite_in_the_Old_Style_Op._28__Nikolai_Kapustin/Suite_in_the_Old_Style_Op._28__Nikolai_Kapustin_2-2.png"
        ),
        MAP_PATH["db_image_images"].joinpath(
            "Suite_in_the_Old_Style_Op._28__Nikolai_Kapustin/Suite_in_the_Old_Style_Op._28__Nikolai_Kapustin_4-4.png"
        ),
        MAP_PATH["db_image_images"].joinpath(
            "Suite_in_the_Old_Style_Op._28__Nikolai_Kapustin/Suite_in_the_Old_Style_Op._28__Nikolai_Kapustin_12-12.png"
        ),
    ],
    "速度记号+踏板": [
        MAP_PATH["db_image_images"].joinpath(
            "Summer_After_Noon_Live/Summer_After_Noon_Live_1-1.png"
        )
    ],
    "踏板": [
        MAP_PATH["db_image_images"].joinpath(
            "Wiege___ALIEN_STAGE/Wiege___ALIEN_STAGE_1-1.png"
        )
    ],
    "连线": [
        MAP_PATH["db_image_images"].joinpath("___7_-__/___7_-___1-1.png"),
        MAP_PATH["db_image_images"].joinpath("___7_-__/___7_-___2-2.png"),
        MAP_PATH["db_image_images"].joinpath(
            "9_Variations_in_C_major_on_the_arietta_Lison_dormait_K._264__Wolfgang_Amadeus_Mozart/9_Variations_in_C_major_on_the_arietta_Lison_dormait_K._264__Wolfgang_Amadeus_Mozart_2-2.png"
        ),
        MAP_PATH["db_image_badcase_complex"].joinpath(
            "Ballade_No.1_in_G_minor_-_not_by_Chopin/Ballade_No.1_in_G_minor_-_not_by_Chopin_7-7.png"
        ),
        MAP_PATH["db_image_badcase_complex"].joinpath(
            "Ballade_No.3_Shortened__Frdric_Chopin__Jaxy/Ballade_No.3_Shortened__Frdric_Chopin__Jaxy_3-3.png"
        ),
        MAP_PATH["db_image_images"].joinpath(
            "Goodbye_Mr._Rachmaninov_by_Cyprien_Katsaris/Goodbye_Mr._Rachmaninov_by_Cyprien_Katsaris_2-2.png"
        ),
    ],
    "延音线": [
        MAP_PATH["db_image_images"].joinpath(
            "Dies_Irae__Giuseppe_Verdi/Dies_Irae__Giuseppe_Verdi_5-5.png"
        ),
        MAP_PATH["db_image_images"].joinpath(
            "Dies_Irae__Giuseppe_Verdi/Dies_Irae__Giuseppe_Verdi_7-7.png"
        ),
        MAP_PATH["db_image_images"].joinpath(
            "Extrapolation_Vienna__xbat302/Extrapolation_Vienna__xbat302_4-4.png"
        ),
        MAP_PATH["db_image_images"].joinpath(
            "Goodbye_Mr._Rachmaninov_by_Cyprien_Katsaris/Goodbye_Mr._Rachmaninov_by_Cyprien_Katsaris_2-2.png"
        ),
        MAP_PATH["db_image_images"].joinpath(
            "Introduction__Rondo_Le_Rossignol_Op.13__zans26fs/Introduction__Rondo_Le_Rossignol_Op.13__zans26fs_1-1.png"
        ),
        MAP_PATH["db_image_images"].joinpath(
            "Scherzo_in_D_minor_No._3/Scherzo_in_D_minor_No._3_1-1.png"
        ),
        # show
        MAP_PATH["db_image_images"].joinpath(
            "Partitura_sem_ttulo__luccaforpastoregmail.com/Partitura_sem_ttulo__luccaforpastoregmail.com_1-1.png"
        ),
    ],
    "声部分配": [
        MAP_PATH["db_image_badcase"].joinpath("moonlight3/moonlight3-01.png"),
        # (3-3)
        MAP_PATH["db_image_images"].joinpath(
            "Partitura_sem_ttulo__luccaforpastoregmail.com/Partitura_sem_ttulo__luccaforpastoregmail.com_1-1.png"
        ),
        # (4-2-2)
        MAP_PATH["db_image_images"].joinpath(
            "Extrapolation_Vienna__xbat302/Extrapolation_Vienna__xbat302_4-4.png"
        ),
        # (3-3,4-2)
    ],
}


# ------------------------------
# io函数
# ------------------------------
def read_dets(p):
    dets = np.loadtxt(p)
    if len(dets.shape) == 1:
        dets = dets[np.newaxis, :]
    return dets


def read_slur_result(p):
    if p.exists():
        res = np.loadtxt(p)
        if len(res.shape) == 1:
            res = res[np.newaxis, :]
    else:
        res = np.zeros((0, 6), dtype=float)
    return res


def read_ocr_labels(p):
    ocr_labels = []
    if p.exists():
        df = pd.read_csv(p, sep='" "', header=None, engine="python")
        df[0] = df[0].str.replace(r'^"', "", regex=True)
        df[5] = df[5].str.replace('"$', "", regex=True).astype(float)
        for i, row in df.iterrows():
            ocr_labels.append([row[0], row[1], row[2], row[3], row[4], row[5]])
    return ocr_labels


def construct_SpSheetMusicBuilder_with_debug_dir(
    debug_dir, p_debug_item=Path("~/link/piano-ai").expanduser()
):
    import sys

    None if str(p_debug_item) in sys.path else sys.path.append(str(p_debug_item))

    from src.imageToXml.sp_musicxml import SpSheetMusicBuilder, LineMask

    debug_dir = Path(debug_dir)
    assert debug_dir.exists()

    return SpSheetMusicBuilder(
        read_dets(debug_dir.joinpath("label_in_image.txt")),
        LineMask(
            np.load(debug_dir.joinpath("mask_int8.npy")).astype(np.int16),
            json.load(open(debug_dir.joinpath("bar_class_all.json"))),
        ),
        ocr_labels=read_ocr_labels(debug_dir.joinpath("ocr_labels.txt")),
        slurs=read_slur_result(debug_dir.joinpath("slur_result.txt")),
    )


def MusicXmlBuilder_generate_xml_output_save(
    self,
    labels_all: List,
    masks_all: np.ndarray,
    bar_class_all: Dict,
    ocr_label_all: List,
    image_all: np.ndarray,
    out_path: str,
    slur_result_all: List,
    img_paths: List[str] = [],
) -> None:
    def handel(
        self,
        labels_all,
        masks_all,
        bar_class_all,
        ocr_label_all,
        slur_result_all,
    ):
        p_debug = Path(self.debug_dir)

        np.savetxt(p_debug.joinpath("label_in_image.txt"), np.array(labels_all))
        np.save(p_debug.joinpath("mask_int8.npy"), masks_all)
        p_debug.joinpath("bar_class_all.json").write_text(json.dumps(bar_class_all))
        with p_debug.joinpath("ocr_labels.txt").open("w") as f:
            for label in ocr_label_all:
                if not label[0]:  # 跳过空文本
                    continue
                f.write(
                    f'"{label[0]}" "{label[1]}" "{label[2]}" "{label[3]}" "{label[4]}" "{label[5]}"\n'
                )
        np.savetxt(p_debug.joinpath("slur_result.txt"), np.array(slur_result_all))

    """生成XML输出文件"""

    from src.imageToXml.main import SpSheetMusicBuilder
    from src.imageToXml.get_position import LineMask

    handel(
        self,
        labels_all,
        masks_all,
        bar_class_all,
        ocr_label_all,
        slur_result_all,
    )

    # 创建LineMask和SpSheetMusicBuilder
    line_mask = LineMask(masks_all, bar_class_all)
    builder = SpSheetMusicBuilder(
        np.array(labels_all),
        line_mask,
        ocr_label_all,
        np.array(slur_result_all),
        base_size=(self.image_width, self.image_height),
    )

    # 生成XML文件
    builder.to_xml(out_path)

    # 调试模式下的额外输出
    if self.is_debug:
        # 保存bar_class_all到JSON文件
        with open(os.path.join(self.debug_dir, "bar_class_all.json"), "w") as f:
            json.dump(bar_class_all, f)
        # 打印照片列表
        for i, img_path in enumerate(img_paths):
            print(f"Page {i}: {img_path}")
        # 绘制标签位置
        self.label_in_image(
            labels_all,
            image_all,
            line_mask,
            os.path.join(self.debug_dir, "box_position.png"),
        )

        # 绘制连接性
        draw_img = image_all.copy()
        draw_img = builder.draw_connectivity(draw_img)
        cv2.imwrite(os.path.join(self.debug_dir, "connectivity.png"), draw_img)


# ------------------------------
# 绘图函数
# ------------------------------
def draw_img_dets(img, dets, mask=None, color=COLORS["red"][2], thickness=3):
    if mask is not None:
        mask = np.array(mask)
        dets = dets[mask]
    return draw_boxes(img, dets[:, 1:5], color=color, thickness=thickness)


def draw_img_stems(
    img, stems, indexs=np.zeros(0, dtype=int), color=COLORS["red"][2], thickness=3
):
    mask = np.array(indexs)
    if mask.size:
        stems = stems[mask, :]

    return draw_boxes(img, stems[:, :4], color=color, thickness=thickness)


def draw_img_ocr(img, ocr_dets, ocr_text, mask=None):
    if mask is not None:
        mask = np.array(mask)
        ocr_dets = ocr_dets[mask]
        ocr_text = ocr_text[mask]

    img = draw_boxes(img, ocr_dets[:, 1:5], color=COLORS["red"][1], thickness=2)
    img = draw_text(
        img,
        move_point(ocr_dets[:, [1, 2]], -50, 0),
        ocr_dets[:, 0].astype(int).astype(str),
        color=COLORS["blue"][5],
        thickness=5,
    )
    img = draw_text(
        img,
        move_point(ocr_dets[:, [1, 2]], 0, 0),
        ocr_text,
        color=COLORS["red"][5],
        thickness=2,
    )
    return img


def draw_direction_boxes(
    img,
    directions,
    color_start=COLORS["red"][2],
    color_stop=COLORS["blue"][2],
    thickness=10,
):
    for direction in directions:
        text = "{} {}".format(
            direction.element_id,
            "{},{},{}".format(direction.rowid, direction.measureid, direction.staff),
        )
        bbox = direction.bbox.astype(int)
        if direction.start_stop == "start":
            bbox = sp.cvpl.move_boxes(bbox[np.newaxis, :], -25)[0]
            cv2.rectangle(
                img, bbox[:2], bbox[2:], color=color_start, thickness=thickness
            )

        if direction.start_stop == "stop":
            bbox = sp.cvpl.move_boxes(bbox[np.newaxis, :], 25)[0]
            cv2.rectangle(
                img, bbox[:2], bbox[2:], color=color_stop, thickness=thickness
            )
        cv2.putText(
            img,
            direction.element_id.split("_")[-1],
            move_point(bbox[:2][np.newaxis, :], -50, 20)[0],
            cv2.FONT_HERSHEY_SIMPLEX,
            2,
            thickness=3,
            color=COLORS["pink"][4],
        )
        cv2.putText(
            img,
            "{},{},{}".format(direction.rowid, direction.measureid, direction.staff),
            bbox[:2],
            cv2.FONT_HERSHEY_SIMPLEX,
            2,
            thickness=4,
            color=COLORS["blue"][6],
        )
    return img


def draw_arrows(img, points, angles):
    def handel(img, x, y, angle):
        import math

        dx = int(round(20 * math.cos(math.radians(angle))))
        dy = int(round(20 * math.sin(math.radians(angle))))
        cv2.arrowedLine(img, (x, y), (x + dx, y + dy), (255, 0, 0), 2, tipLength=0.3)

    points = points.astype(int)
    for (x, y), angle in zip(points, angles):
        handel(img, x, y, angle)
    return img


def draw_img_xml_note_group(
    img, xml_note_group, notes, stems, color=COLORS["red"][2], thickness=3
):
    groups = xml_note_group.groups
    img = draw_img_dets(img, notes, groups[:, 0], color=color, thickness=thickness)
    img = draw_img_stems(img, stems, groups[:, 1], color=color, thickness=thickness)
    return img


def draw_site(img, smv):
    COLORS_CYCLE.reset()
    for i in np.unique(smv.dets_bits[:, 6]):
        color = next(COLORS_CYCLE)
        img = draw_img_dets(img, smv.dets_bits, smv.dets_bits[:, 6] == i, color=color)
        img = draw_text(
            img,
            move_point(
                smv.dets_bits[smv.dets_bits[:, 6] == i][:, [1, 2]][[0], :],
                0,
                -20,
            ),
            np.array([i]).astype(int).astype(str),
            fontscale=1,
        )
    img = draw_text(
        img,
        move_point(smv.dets_bits[:, [1, 2]], 0, 10),
        smv.dets_bits[:, 0].astype(int).astype(str),
        fontscale=1,
        thickness=2,
        color=COLORS["blue"][2],
    )

    # # 不可见bit
    # df_plot = smv.dets_bits[~smv.dets_bits.visible]
    # if df_plot.size:
    #     img = draw_img_dets(
    #         img,
    #         df_plot.to_numpy(),
    #         color=COLORS["gray"][5], thickness=2
    #     )
    return img


def draw_xml_groups(img, xml_groups):
    boxes = np.c_[[g.boundbox for g in xml_groups]].astype(int)
    if boxes.size:
        img = draw_boxes(img, boxes, thickness=2)
        img = draw_text(
            img,
            boxes[:, [0, 1]],
            np.arange(len(xml_groups)).astype(str),
            fontscale=1,
            thickness=2,
        )
    else:
        print("boxes is empty")
    return img


def draw_note_stem(img, notes, stems):
    COLORS_CYCLE.reset()
    img = draw_img_stems(img, stems, color=next(COLORS_CYCLE), thickness=2)
    img = draw_img_dets(img, notes, color=next(COLORS_CYCLE), thickness=2)
    next(COLORS_CYCLE)
    img = draw_text(
        img,
        move_point(stems[:, [0, 1]], -10),
        np.arange(stems.shape[0]).astype(str),
        fontscale=0.5,
        color=next(COLORS_CYCLE),
        thickness=2,
    )
    next(COLORS_CYCLE)
    img = draw_text(
        img,
        move_point(notes[:, [1, 2]], 0, -10),
        np.arange(notes.shape[0]).astype(str),
        fontscale=0.5,
        color=next(COLORS_CYCLE),
        thickness=2,
    )
    return img


def draw_tuplet(img, smv, stems, numbers):
    def handel_stem_box(arr_number2stem, number_id):
        box = stems[arr_number2stem[arr_number2stem[:, 0] == number_id, 1]]
        return np.concatenate([box[:, [0, 1]].min(axis=0), box[:, [2, 3]].max(axis=0)])

    if not smv.tuplets_source.size:
        return img

    color = COLORS["purple"][2]
    arr_number2stem = np.array(
        [[d["number_id"], d["stem_id"]] for d in smv.tuplets_source]
    )
    if arr_number2stem.size:
        arr_number2stem = np.unique(arr_number2stem, axis=0)

        dets_numbers_stem = (
            np.c_[
                np.unique(arr_number2stem[:, 0]),
                np.array(
                    [
                        handel_stem_box(arr_number2stem, number_id)
                        for number_id in np.unique(arr_number2stem[:, 0])
                    ]
                ),
            ]
            .round()
            .astype(int)
        )
        number_ids = [
            i for i in np.unique(arr_number2stem[:, 0]) if i < numbers.shape[0]
        ]
        if number_ids:
            img = draw_img_dets(img, numbers[number_ids], color=color)
        img = draw_img_dets(img, dets_numbers_stem, color=color)
    return img


def draw_metadata(img, smv):
    if len(smv.metadata.keys()) == 0:
        return img
    # 该小节外接框
    texts = np.array(["{}:{}".format(k, v) for k, v in smv.metadata.items()])
    points = smv.get_measure_boundbox()[[0, 1]]
    points = points[np.newaxis, :][np.repeat(0, texts.size)]
    points[:, 1] = points[:, 1] + (np.arange(texts.size) + 1) * 20

    img = draw_text(
        img,
        points,
        texts,
        fontscale=0.6,
        thickness=2,
        color=COLORS["yellow"][4],
    )
    return img


def draw_voice(img, smv):
    stem2points = np.c_[
        smv.dets_bits[:, [9]],
        smv.dets_bits[:, [1, 3]].mean(axis=1).round().astype(int),
        smv.dets_bits[:, [2, 4]].mean(axis=1).round().astype(int),
    ]  # [steid_measure, cx , cy]

    stem2voice = np.apply_along_axis(
        lambda arr: [
            arr[4],
            smv.time_values_group[smv.time_values_group[:, 0] == arr[3], 3][0],
        ],
        axis=1,
        arr=smv.group_bit,
    )
    stem2voice = stem2voice[stem2voice[:, 1] > 0]

    COLORS_CYCLE.reset()
    for v in np.unique(stem2voice[:, 1]):
        steid_measures = stem2voice[stem2voice[:, 1] == v, 0]
        points = stem2points[np.isin(stem2points[:, 0], steid_measures), 1:]
        color_point = next(COLORS_CYCLE)
        color_line = next(COLORS_CYCLE)
        if points.shape[0] > 1:
            draw_points_relations(
                img,
                points[:-1, :],
                points[1:, :],
                radius=10,
                color_point1=color_point,
                color_point2=color_point,
                color_line=color_line,
            )
        else:
            draw_points_relations(
                img,
                points,
                points,
                radius=10,
                color_point1=color_point,
                color_point2=color_point,
                color_line=color_line,
            )

    # start_time time
    img = draw_img_dets(img, smv.dets_group)
    img = draw_text(
        img,
        move_point(smv.dets_group[:, [1, 2]], 0, 0),
        smv.dets_group[:, 0].astype(int).astype(str),
        thickness=2,
        fontscale=1,
    )
    img = draw_text(
        img,
        move_point(smv.dets_group[:, [1, 2]], 0, 20),
        smv.time_values_group[:, 1].astype(str),
        thickness=2,
        fontscale=0.8,
        color=COLORS["blue"][4],
    )
    img = draw_text(
        img,
        move_point(smv.dets_group[:, [1, 2]], 0, 40),
        smv.time_values_group[:, 2].astype(str),
        thickness=2,
        fontscale=0.8,
        color=COLORS["blue"][4],
    )
    return img


# ------------------------------
# 其他函数
# ------------------------------


def copy_img_to_badcase(p):
    p = Path(p)
    assert p.suffix in [".png", ".jpg"]
    p_badcase = Path("/home/algorithm_team/badcase/images")
    import shutil

    shutil.copy(p, p_badcase.joinpath(p.name))
    assert p_badcase.joinpath(p.name).exists()
    print("[copy] {} to badcase".format(p.name))


def copy_pdf_png_to_each_debug_dir(p_debug):
    p_debug = Path(p_debug)
    info_debug = ut.df.iter_dir(p_debug, select="d")
    info_debug.index = info_debug["name"]
    info_imgs = ut.df.iter_dir(info_debug.at["pdf", "path"])

    info_imgs = ut.df.iter_dir(info_debug.at["pdf", "path"])
    info_imgs.index = info_imgs["path"].apply(lambda x: x.stem)

    for row in info_debug.itertuples():
        if row.Index in info_imgs.index:
            shutil.copy(
                info_imgs.at[row.Index, "path"], row.path.joinpath("00_source.png")
            )


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


# ------------------------------
# 单元测试
# ------------------------------


class UnitTestFunction:
    """
    单元测试比较函数
    """

    def _compare_ndarray_shape(self, arr1, arr2):
        if not arr1.ndim == arr2.ndim:
            return False
        if not arr1.shape == arr2.shape:
            return False
        return True

    def compare_ndarray(self, arr1, arr2):
        flag = False
        if self._compare_ndarray_shape(arr1, arr2):
            if np.isclose(arr1, arr2, 0.01).all():
                flag = True
        return flag


class UnitTestBase:
    """
    单元测试基类
    save_input_and_output       用于保存输入和输出
    unit_test                   进行单元测试
    tfunc                       比较函数类
    """

    tfunc = UnitTestFunction()

    def __init__(self, p_ut):
        self.p_ut = Path(p_ut)

    def _init_dir(self, name):
        p_ut = self.p_ut
        p_input = p_ut.joinpath("input").joinpath("{}.pkl".format(name))
        p_output = p_ut.joinpath("gt").joinpath("{}.pkl".format(name))
        p_input.parent.mkdir(parents=True, exist_ok=True)
        p_output.parent.mkdir(parents=True, exist_ok=True)
        return p_input, p_output

    @staticmethod
    def pickle_dump(p_input, **kvargs):
        with open(p_input, "wb") as f:
            pickle.dump(kvargs, f)
        print("[save] {} in {}".format(p_input.name, p_input.parent))

    @staticmethod
    def pickle_loads(p_input):
        data = {}
        with open(p_input, "rb") as f:
            data = pickle.load(f, fix_imports=True, encoding="bytes")
        return data

    @abstractmethod
    def _save_input(self, p_debug_item, p_input):
        pass

    @abstractmethod
    def process(self, **kvargs):
        pass

    @abstractmethod
    def save_input_and_output(self, p_debug_item):
        pass

        # p_debug_item = Path(p_debug_item)
        # assert p_debug_item.exists()

        # p_input, p_output = self._init_dir(p_debug_item.name)
        # self._save_input(p_debug_item, p_input)
        # data_input = self.pickle_loads(p_input)
        # data_ouput = self.process(**data_input)
        # self.pickle_dump(p_output,
        #       # 指定属性名
        #       XXX=data_ouput)

    @abstractmethod
    def unit_test(self, item):
        pass
        # p_input, p_output = self._init_dir(item)
        # data_input = self.pickle_loads(p_input)
        # data_tartget = self.pickle_loads(p_output)
        # XXX = self.process(**data_input)
        # 判定是否pass
        # assert self.tfunc.func(
        #       data_tartget["XXX"],
        #       XXX),"[not pass] {}".format(item)


class UnitTestSlurLoc(UnitTestBase):
    """
    连线定位, 符头符杆匹配
    """

    def __init__(self, p_ut=P_UNIT_TEST.joinpath("slur_loc")):
        super().__init__(p_ut)

    def _save_input(self, p_debug_item, p_input):
        builder = construct_SpSheetMusicBuilder_with_debug_dir(p_debug_item)
        self.pickle_dump(
            p_input,
            slurs=read_slur_result(p_debug_item.joinpath("slur_result.txt")),
            dets=builder.dets,
            stems=builder.stems,
            notes=builder.notes,
            note2stem_items=builder.note2stem_items,
        )

    def process(self, **kvargs):
        slurs = kvargs["slurs"]
        dets = kvargs["dets"]
        stems = kvargs["stems"]
        notes = kvargs["notes"]
        note2stem_items = kvargs["note2stem_items"]

        from src.imageToXml.SpNotationsSlur import SlurParser
        from src.imageToXml.SpNotationsSlur import SlurPossibility
        from src.imageToXml.SpNotationsSlur import utils

        slurs = SlurParser.process_slurs(slurs)
        # 符头符杆视为整体
        stem_notes_dets = utils.get_stem_notes_dets(stems, notes, note2stem_items)
        # 音位连通性
        bit_connected = SlurParser.get_bit_connected(stem_notes_dets)
        # ------------------------------
        # 构造 slurs_group
        #   一项代表一条线
        #   slur 的两个点匹配符杆符头
        # ------------------------------
        # slurs_group: [
        #   index,possibility
        #   match_possibility_start,match_possibility_stop
        #   stem_id_start, stem_id_stop,
        #   note_id_start,note_id_stop
        #   start_is_row_start, stop_is_row_stop
        # ]
        slurs_group = np.c_[
            np.arange(slurs.shape[0]),
            np.repeat(SlurPossibility.UNKNOW.value, slurs.shape[0]),
        ]
        # slurs 匹配符头符杆
        slurs_group = np.hstack(
            [
                slurs_group,
                np.apply_along_axis(
                    lambda arr: SlurParser.slur_point_match_stem(
                        arr, stem_notes_dets, stems, bit_connected
                    ),
                    axis=1,
                    arr=slurs,
                ),
            ]
        )
        # 保险：去除未匹配符杆的项
        slurs_group = slurs_group[(slurs_group[:, 4] >= 0) & (slurs_group[:, 5] >= 0)]
        # slurs 匹配符头
        slurs_group = np.hstack(
            [
                slurs_group,
                np.vstack(
                    [
                        np.apply_along_axis(
                            lambda arr: SlurParser.slur_point_match_note(
                                slurs[arr[0], [0, 1]], arr[1], notes, note2stem_items
                            ),
                            axis=1,
                            arr=slurs_group[:, [0, 4]],
                        ),
                        np.apply_along_axis(
                            lambda arr: SlurParser.slur_point_match_note(
                                slurs[arr[0], [3, 4]], arr[1], notes, note2stem_items
                            ),
                            axis=1,
                            arr=slurs_group[:, [0, 5]],
                        ),
                    ]
                ).T,
            ]
        )
        # 排序
        slurs_group = slurs_group[
            np.lexsort(
                [
                    notes[slurs_group[:, 3], 1],  # x1
                    utils.get_notes_measure_index(notes[slurs_group[:, 6]]),  # 小节位置
                ]
            )
        ]
        return slurs_group

    def save_input_and_output(self, p_debug_item):
        p_debug_item = Path(p_debug_item)
        assert p_debug_item.exists()

        p_input, p_output = self._init_dir(p_debug_item.name)
        self._save_input(p_debug_item, p_input)
        data_input = self.pickle_loads(p_input)
        data_ouput = self.process(**data_input)
        self.pickle_dump(p_output, slurs_group=data_ouput)

    def unit_test(self, item):
        p_input, p_output = self._init_dir(item)
        data_input = self.pickle_loads(p_input)
        data_tartget = self.pickle_loads(p_output)
        slurs_group = self.process(**data_input)
        # 判定是否pass
        assert self.tfunc.compare_ndarray(
            data_tartget["slurs_group"], slurs_group
        ), "[not pass] {}".format(item)
