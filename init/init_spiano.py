from utils_py.general import *
import utils_py as ut
import itertools
import shutil
import sys
import warnings
from typing import List, Dict

from utils_py import spiano as sp
from utils_py.spiano import letterbox, musicxml_to_png, view_resutl
from utils_py.spiano.cvplot import plt_imshow, plt
from utils_py.spiano.cvplot import draw_boxes, draw_text, draw_points
from utils_py.spiano.cvplot import draw_boxes_relations, draw_points_relations
from utils_py.spiano.cvplot import move_boxes, move_point, points2boxes, format_location

from utils_py.spiano.cvplot import COLORS,COLORS_CYCLE

DICT_SLUR_COLOR = {
    "point_start": COLORS["red"][1],
    "point_stop": COLORS["cyan"][1],
    "line": COLORS["green"][3],
    "stem_start": COLORS["red"][3],
    "stem_stop": COLORS["cyan"][3],
    "note_start": COLORS["red"][5],
    "note_stop": COLORS["cyan"][5],
}


p_temp = str(Path("~/link/piano-ai").expanduser())
os.chdir(p_temp)
None if p_temp in sys.path else sys.path.append(p_temp)
del p_temp




cv2 = sp.cv2


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
        df[0] = df[0].str.replace(r'^"', '', regex=True)
        df[5] = df[5].str.replace('"$', '', regex=True).astype(float)
        for i, row in df.iterrows():
            ocr_labels.append([
                row[0], row[1], row[2], row[3], row[4], row[5]
            ])
    return ocr_labels


def construct_SpSheetMusicBuilder_with_debug_dir(debug_dir, p_root=Path("~/link/piano-ai").expanduser()):
    import sys
    None if str(p_root) in sys.path else sys.path.append(str(p_root))

    from src.imageToXml.sp_musicxml import SpSheetMusicBuilder, LineMask

    debug_dir = Path(debug_dir)
    assert debug_dir.exists()

    return SpSheetMusicBuilder(
        read_dets(debug_dir.joinpath("label_in_image.txt")),
        LineMask(np.load(debug_dir.joinpath("mask_int8.npy")).astype(np.int16),
                 json.load(open(debug_dir.joinpath("bar_class_all.json")))),
        ocr_labels=read_ocr_labels(debug_dir.joinpath("ocr_labels.txt")),
        slurs=read_slur_result(debug_dir.joinpath("slur_result.txt")))




def MusicXmlBuilder_generate_xml_output_save(
    self,
    labels_all: List,
    masks_all: np.ndarray,
    bar_class_all: Dict,
    ocr_label_all: List,
    image_all: np.ndarray,
    out_path: str,
    slur_result_all: List,
    img_paths: List[str] = []
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

        np.savetxt(p_debug.joinpath("label_in_image.txt"),
                   np.array(labels_all))
        np.save(p_debug.joinpath("mask_int8.npy"), masks_all)
        p_debug.joinpath("bar_class_all.json").write_text(
            json.dumps(bar_class_all))
        with p_debug.joinpath("ocr_labels.txt").open("w") as f:
            for label in ocr_label_all:
                if not label[0]:  # 跳过空文本
                    continue
                f.write(
                    f'"{label[0]}" "{label[1]}" "{label[2]}" "{label[3]}" "{label[4]}" "{label[5]}"\n')
        np.savetxt(p_debug.joinpath("slur_result.txt"),
                   np.array(slur_result_all))

    """生成XML输出文件"""

    
    from src.imageToXml.main import SpSheetMusicBuilder
    from src.imageToXml.get_position import LineMask

    handel(self,
           labels_all,
           masks_all,
           bar_class_all,
           ocr_label_all,
           slur_result_all,)

    # 创建LineMask和SpSheetMusicBuilder
    line_mask = LineMask(masks_all, bar_class_all)
    builder = SpSheetMusicBuilder(np.array(labels_all), line_mask, ocr_label_all, np.array(slur_result_all),
                                  base_size=(self.image_width, self.image_height))

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
            os.path.join(self.debug_dir, "box_position.png")
        )

        # 绘制连接性
        draw_img = image_all.copy()
        draw_img = builder.draw_connectivity(draw_img)
        cv2.imwrite(os.path.join(self.debug_dir, 'connectivity.png'), draw_img)


def draw_img_dets(img, dets, mask=None, color=COLORS["red"][2], thickness=3):
    if mask is not None:
        mask = np.array(mask)
        dets = dets[mask]
    return draw_boxes(img, dets[:, 1:5], color=color, thickness=thickness)


def draw_img_stems(img, stems, indexs=np.zeros(0, dtype=int),
                   color=COLORS["red"][2], thickness=3):
    mask = np.array(indexs)
    if mask.size:
        stems = stems[mask, :]

    return draw_boxes(img, stems[:, :4], color=color, thickness=thickness)


def draw_img_ocr(img, ocr_dets, ocr_text, mask=None):
    if mask is not None:
        mask = np.array(mask)
        ocr_dets = ocr_dets[mask]
        ocr_text = ocr_text[mask]

    img = draw_boxes(img, ocr_dets[:, 1:5],
                     color=COLORS["red"][1], thickness=2)
    img = draw_text(img, move_point(ocr_dets[:, [1, 2]], -50, 0),
                    ocr_dets[:, 0].astype(int).astype(str),
                    color=COLORS["blue"][5], thickness=5
                    )
    img = draw_text(img, move_point(ocr_dets[:, [1, 2]], 0, 0),
                    ocr_text, color=COLORS["red"][5], thickness=2)
    return img


def draw_direction_boxes(img, directions, color_start=COLORS["red"][2],
                         color_stop=COLORS["blue"][2], thickness=10):
    for direction in directions:
        text = "{} {}".format(direction.element_id,
                              "{},{},{}".format(direction.rowid, direction.measureid, direction.staff))
        bbox = direction.bbox.astype(int)
        if direction.start_stop == "start":
            bbox = sp.cvpl.move_boxes(bbox[np.newaxis, :], -25)[0]
            cv2.rectangle(img, bbox[:2], bbox[2:],
                          color=color_start, thickness=thickness)

        if direction.start_stop == "stop":
            bbox = sp.cvpl.move_boxes(bbox[np.newaxis, :], 25)[0]
            cv2.rectangle(img, bbox[:2], bbox[2:],
                          color=color_stop, thickness=thickness)
        cv2.putText(
            img, direction.element_id.split("_")[-1],
            move_point(bbox[:2][np.newaxis, :], -50,
                       20)[0], cv2.FONT_HERSHEY_SIMPLEX, 2,
            thickness=3, color=COLORS["pink"][4]
        )
        cv2.putText(
            img, "{},{},{}".format(
                direction.rowid, direction.measureid, direction.staff),
            bbox[:2],
            cv2.FONT_HERSHEY_SIMPLEX, 2,
            thickness=4, color=COLORS["blue"][6]
        )
    return img


def draw_arrows(img, points, angles):
    def handel(img, x, y, angle):
        import math
        dx = int(round(20 * math.cos(math.radians(angle))))
        dy = int(round(20 * math.sin(math.radians(angle))))
        cv2.arrowedLine(img, (x, y), (x + dx, y + dy),
                        (255, 0, 0), 2, tipLength=0.3)
    points = points.astype(int)
    for (x, y), angle in zip(points, angles):
        handel(img, x, y, angle)
    return img


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
    info_debug = ut.df.iter_dir(p_debug,select="d")
    info_debug.index = info_debug["name"]
    info_imgs = ut.df.iter_dir(info_debug.at["pdf","path"])

    info_imgs = ut.df.iter_dir(info_debug.at["pdf","path"])
    info_imgs.index = info_imgs["path"].apply(lambda x:x.stem)

    for row in info_debug.itertuples():
        if row.Index in info_imgs.index:
            shutil.copy(info_imgs.at[row.Index,"path"],
                        row.path.joinpath("00_source.png"))



P_DB_IMAGE = Path("/home/lcc/link/share/db_image")
MAP_PATH = {
    "db_image": P_DB_IMAGE,
    "db_image_images": P_DB_IMAGE.joinpath("images"),
    "db_image_badcase": Path(""),
    "db_image_badcase_complex": P_DB_IMAGE.joinpath("complex_music_images_4090_badcase"),
    "db_image_pos": P_DB_IMAGE.joinpath("PositiveSampleImages"),
    "db_image_pdf_imslp": Path(""),
    "workspace": Path(""),
    "artificial": P_DB_IMAGE.joinpath("artificial_muxicxml"),
}


TEST_SET = {
    "虚实线": [
        MAP_PATH["artificial"].joinpath("line_group_qinghuaci-1.png"),
        MAP_PATH["artificial"].joinpath(
            "lined_strength_qinghuaci_1-1.png"),
        MAP_PATH["artificial"].joinpath("pedal_qinghuaci_1-1.png"),
        MAP_PATH["artificial"].joinpath("lined_speed_qinghuaci_1-1.png"),
        MAP_PATH["artificial"].joinpath("lined_strength_debug_1.png"),
        MAP_PATH["artificial"].joinpath("line_soild_qinghuaci_1.png"),
        MAP_PATH["artificial"].joinpath("qinghuaci-1.png"),
    ],
    "高低八度": [
        MAP_PATH["db_image_images"].joinpath(
            "Liszt_-_Ballade_No._1_in_Db_Major_S._170/Liszt_-_Ballade_No._1_in_Db_Major_S._170_3-3.png"),
        MAP_PATH["db_image_images"].joinpath(
            "Liszt_-_Ballade_No._1_in_Db_Major_S._170/Liszt_-_Ballade_No._1_in_Db_Major_S._170_11-11.png"),
        MAP_PATH["db_image_images"].joinpath(
            "Liszt_-_Ballade_No._1_in_Db_Major_S._170/Liszt_-_Ballade_No._1_in_Db_Major_S._170_13-13.png"),
        MAP_PATH["db_image_images"].joinpath(
            "Scherzo_No.1_Op.39/Scherzo_No.1_Op.39_3-3.png"),
        MAP_PATH["db_image_images"].joinpath(
            "Turkish_March_-_Volodos/Turkish_March_-_Volodos-9.png"),
        # MAP_PATH["db_image_badcase"].joinpath(
        #     "694071617332031488_0.jpg"),
    ],


    "跳房子": [
        MAP_PATH["db_image"].joinpath("line_group_qinghuaci-1.png"),
        MAP_PATH["db_image_images"].joinpath(
            "Suite_in_the_Old_Style_Op._28__Nikolai_Kapustin/Suite_in_the_Old_Style_Op._28__Nikolai_Kapustin_2-2.png"),
        MAP_PATH["db_image_images"].joinpath(
            "Suite_in_the_Old_Style_Op._28__Nikolai_Kapustin/Suite_in_the_Old_Style_Op._28__Nikolai_Kapustin_4-4.png"),
        MAP_PATH["db_image_images"].joinpath(
            "Suite_in_the_Old_Style_Op._28__Nikolai_Kapustin/Suite_in_the_Old_Style_Op._28__Nikolai_Kapustin_12-12.png"),
    ],
    "速度记号+踏板": [
        MAP_PATH["db_image_images"].joinpath(
            "Summer_After_Noon_Live/Summer_After_Noon_Live_1-1.png")
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
            "9_Variations_in_C_major_on_the_arietta_Lison_dormait_K._264__Wolfgang_Amadeus_Mozart/9_Variations_in_C_major_on_the_arietta_Lison_dormait_K._264__Wolfgang_Amadeus_Mozart_2-2.png"),
        # MAP_PATH["db_image_badcase_complex"].joinpath(
        #     "Ballade_No.1_in_G_minor_-_not_by_Chopin/Ballade_No.1_in_G_minor_-_not_by_Chopin_7-7.png"),
        # MAP_PATH["db_image_badcase_complex"].joinpath(
        #     "Ballade_No.3_Shortened__Frdric_Chopin__Jaxy/Ballade_No.3_Shortened__Frdric_Chopin__Jaxy_3-3.png"),
        MAP_PATH["db_image_images"].joinpath(
            "Goodbye_Mr._Rachmaninov_by_Cyprien_Katsaris/Goodbye_Mr._Rachmaninov_by_Cyprien_Katsaris_2-2.png"),
        # MAP_PATH["db_image_pos"].joinpath(
        #     "29ecc284-8a1e-43fa-ab9b-7fc2cdb9f5cc.jpg"),

        # # 连线相交 导致 起点终点异常
        # MAP_PATH["db_image_images"].joinpath(
        #     "Ballade_No.1_in_G_minor_-_not_by_Chopin/Ballade_No.1_in_G_minor_-_not_by_Chopin_2-2.png"),
        # # 声部跨五线
        # MAP_PATH["db_image_images"].joinpath(
        #     "Goodbye_Mr._Rachmaninov_by_Cyprien_Katsaris/Goodbye_Mr._Rachmaninov_by_Cyprien_Katsaris_4-4.png"),
        # MAP_PATH["db_image_images"].joinpath(
        #     "Liszt_-_Ballade_No._1_in_Db_Major_S._170/Liszt_-_Ballade_No._1_in_Db_Major_S._170_10-10.png"),
        # MAP_PATH["db_image_images"].joinpath(
        #     "La_Campanella_Caprice_Op.411__Wilhelm_Taubert/La_Campanella_Caprice_Op.411__Wilhelm_Taubert_1-1.png"),
        # # 连线交错
        # MAP_PATH["db_image_images"].joinpath(
        #     "Liszt_-_Ballade_No._1_in_Db_Major_S._170/Liszt_-_Ballade_No._1_in_Db_Major_S._170_12-12.png"),

        # # 连线模型识别结果差
        # MAP_PATH["db_image_pdf_imslp"].joinpath("296892.pdf"),
        # MAP_PATH["db_image_pdf_imslp"].joinpath("365669.pdf"),
        # MAP_PATH["db_image_pdf_imslp"].joinpath("468553.pdf"),
        # MAP_PATH["db_image_pdf_imslp"].joinpath("699066.pdf"),
        # MAP_PATH["db_image_pdf_imslp"].joinpath("719134.pdf"),
        # MAP_PATH["db_image_pdf_imslp"].joinpath("797433.pdf"),
        # MAP_PATH["db_image_pdf_imslp"].joinpath("823128.pdf"),

    ],
    "延音线": [
        MAP_PATH["db_image_images"].joinpath(
            "Dies_Irae__Giuseppe_Verdi/Dies_Irae__Giuseppe_Verdi_5-5.png"),
        MAP_PATH["db_image_images"].joinpath(
            "Dies_Irae__Giuseppe_Verdi/Dies_Irae__Giuseppe_Verdi_7-7.png"),
        MAP_PATH["db_image_images"].joinpath(
            "Extrapolation_Vienna__xbat302/Extrapolation_Vienna__xbat302_4-4.png"),
        MAP_PATH["db_image_images"].joinpath(
            "Goodbye_Mr._Rachmaninov_by_Cyprien_Katsaris/Goodbye_Mr._Rachmaninov_by_Cyprien_Katsaris_2-2.png"),
        MAP_PATH["db_image_images"].joinpath(
            "Introduction__Rondo_Le_Rossignol_Op.13__zans26fs/Introduction__Rondo_Le_Rossignol_Op.13__zans26fs_1-1.png"),
        MAP_PATH["db_image_images"].joinpath(
            "Scherzo_in_D_minor_No._3/Scherzo_in_D_minor_No._3_1-1.png"),
        # show
        MAP_PATH["db_image_images"].joinpath(
            "Partitura_sem_ttulo__luccaforpastoregmail.com/Partitura_sem_ttulo__luccaforpastoregmail.com_1-1.png"),
    ],
    "声部分配": [

        MAP_PATH["db_image_badcase"].joinpath("moonlight3/moonlight3-01.png"),
        # (3-3)
        MAP_PATH["db_image_images"].joinpath(
            "Partitura_sem_ttulo__luccaforpastoregmail.com/Partitura_sem_ttulo__luccaforpastoregmail.com_1-1.png"),
        # (4-2-2)
        MAP_PATH["db_image_images"].joinpath(
            "Extrapolation_Vienna__xbat302/Extrapolation_Vienna__xbat302_4-4.png"),
        # (3-3,4-2)
    ]
}
