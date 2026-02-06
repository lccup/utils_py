"""
# general

utils_py 的基础, 公共的包在genral中导入

> variable
    rng  numpy的随机数生成器

> function
    module_exists
    subset_dict
    rm_rf
    str_step_insert
    time_tag

    [show]
    show_str_arr
    show_obj_attr
    show_dict_key

> class
    Block

"""

import os
from pathlib import Path
import platform
import json as json
import time
from functools import reduce
from collections import namedtuple
import inspect
import pickle

import numpy as np
import pandas as pd

rng = np.random.default_rng()
system = platform.system()


def module_exists(module_name):
    import importlib.util
    import sys

    return module_name in sys.modules or importlib.util.find_spec(module_name)


def subset_dict(data, keep_item=[], regex=None, keep_order=False):
    if isinstance(keep_item, str):
        keep_item = [keep_item]

    data = data.copy()
    keys = []
    if regex:
        keys = pd.Series(list(data.keys()))
        keys = keys[keys.str.match(regex)]
    keys = pd.Series(np.concatenate([keys, keep_item])).unique()

    if keep_order:
        return {k: v for k, v in data.items() if k in keys}
    return {k: data[k] for k in keys}


def update_dict(data, update=None, **kvargs):
    """
    更新字典，覆盖顺序 kvargs > update > data
    """
    data = data.copy()
    if isinstance(update, dict):
        data.update(update)
    data.update(kvargs)
    return data


def handle_type_to_list(data, t=str):
    """
    若 data 为 t 则返回 [data]
    若 data 为list 则返回 data
    """
    assert isinstance(data, (list, t)), "[Error] data is not a list or {}".format(t)
    return [data] if isinstance(data, t) else data


def time_tag(format="%y%m%d_%H%M%S"):
    return time.strftime(format, time.localtime())


##################################################
# show
##################################################


def show_str_arr(data, n_cols=4, order="F", return_str=False):
    """
    Examples
    --------

    import numpy as np
    rng = np.random.default_rng()

    show_str_arr(['12345678910'[:_]
        for _ in rng.integers(1,10,13)],n_cols=3)

    # | 12345678| 1234567  | 123456|
    # | 1234    | 12345678 | 1234  |
    # | 123     | 123456789| 1234  |
    # | 123     | 1        |       |
    # | 123456  | 1        |       |
    """
    data = np.array(data)
    if data.size % n_cols != 0:
        data = np.concatenate([data, np.repeat("", n_cols - data.size % n_cols)])
    data = pd.Series(data)
    data = pd.DataFrame(np.reshape(data, (data.size // n_cols, n_cols), order=order))
    for k in data.columns:
        data[k] = data[k].str.ljust(data[k].str.len().max())
    res = "\n".join(
        data.apply(lambda row: "| {}|".format("| ".join(row.values)), axis=1)
    )
    print(res)

    if return_str:
        return res


def show_obj_attr(
    obj,
    regex_filter=["^_"],
    regex_select=[],
    only="method",
    show_n_cols=3,
    group=False,
    extract_pat="^([^_]+)_",
    return_frame=False,
):

    def handel(value):
        if inspect.ismethod(value):
            return "method"
        if inspect.isfunction(value):
            return "function"
        return "attribute"

    if isinstance(regex_filter, str):
        regex_filter = [regex_filter]
    if isinstance(regex_select, str):
        regex_select = [regex_select]

    df = pd.Series([[name, value] for name, value in inspect.getmembers(obj)]).apply(
        pd.Series
    )
    df.columns = ["name", "value"]
    df["type"] = df["value"].apply(lambda v: handel(v))
    df = df.loc[:, ["name", "type"]]

    # only
    if only:
        assert only in df["type"].unique(), "only must be one of {}".format(
            df["type"].unique()
        )
        df = df[df["type"] == only]

    # filter
    if len(regex_filter) == 1:
        df = df[~df["name"].str.match(regex_filter[0])]
    elif len(regex_filter) == 0:
        pass
    else:
        df = df[
            reduce(lambda x, y: x & y, [~df["name"].str.match(_) for _ in regex_filter])
        ]
    # select
    if len(regex_select) == 1:
        df = df[df["name"].str.match(regex_select[0])]
    elif len(regex_select) == 0:
        pass
    else:
        df = df[
            reduce(lambda x, y: x | y, [df["name"].str.match(_) for _ in regex_select])
        ]

    df = df.sort_values("name").reset_index(drop=True)

    if show_n_cols:
        show_str_arr(df["name"], show_n_cols)
    if group:
        print("[group]".ljust(15, "-"))
        print(
            df["name"].str.extract(extract_pat, expand=False).value_counts(dropna=False)
        )

    if return_frame:
        return df


def show_dict_key(data, tag="", sort_key=True):
    print("> {}['']".format(tag).ljust(75, "-"))
    ks = list(data.keys())
    if sort_key:
        ks = np.sort(ks)
    print(*["  {}".format(k) for k in ks], sep="\n")


def show_memory_size(obj, precision=2):
    module_exists("pympler"), "pip install pympler"

    from pympler.asizeof import asizeof

    def format_memory_size(size_bytes, precision=2):
        """
        将字节大小转换为易读的格式

        Args:
            size_bytes (int): 字节大小
            precision (int): 小数位数精度

        Returns:
            str: 格式化后的内存大小字符串
        """
        # 定义单位
        units = ["B", "KB", "MB", "GB", "TB", "PB"]

        # 处理零或负数
        if size_bytes <= 0:
            return f"0 {units[0]}"

        # 计算合适的单位
        import math

        unit_index = min(int(math.log(size_bytes, 1024)), len(units) - 1)

        # 计算对应单位的大小
        size_in_unit = size_bytes / (1024**unit_index)

        # 格式化输出
        if unit_index == 0:  # 字节，不需要小数
            return f"{int(size_in_unit)} {units[unit_index]}"
        else:
            return f"{size_in_unit:.{precision}f} {units[unit_index]}"

    size_bytes = asizeof(obj)
    print(format_memory_size(size_bytes, precision))
    return size_bytes


##################################################
#  Block
#
# `块` 用于将代码分块
#
# 因为在notebook中用注释将代码分块，无法实现代码的折叠，使得代码较为混乱
#
# 故通过`with Block():`组合构造出可以折叠的with语句块，进而提高代码的可读性
#
# + `with Block():`内部并未与外部进行隔离
#
# + 实现了计时功能
# + 实现了上下文功能
##################################################


class Block:
    """
    用于在notebook中给代码划分区域(块),从而使代码能够折叠

    Examples
    --------


    # 上下文功能
    with Block('context',context={
        'a':'Block','b':':','c':'Hello World'
    }) as context:
        print('inner')
        print('\t',' '.join(context.context.values()))
        print('\t',context.a,context.b,context.c)
    # output
    ## inner
    ## 	 Block : Hello World
    ## 	 Block : Hello World

    # 计时功能
    import time
    with Block('test',show_comment=True):
        print('inner')
        time.sleep(2)
    # output
    ## [start][test] 0509-00:20:47------------------------------------------------
    ## inner
    ## [end][test] 0509-00:20:49--------------------------------------------------
            2.002 s used
    """

    def __init__(self, comment, show_comment=False, context=None):
        self.comment = comment
        self.show_comment = show_comment
        self.context = context
        self.time = 0

    def __enter__(self):
        if self.show_comment:
            self.time = time.time()
            print(
                "[start][{}] {}".format(
                    self.comment, time.strftime("%m%d-%H:%M:%S", time.localtime())
                ).ljust(75, "-")
            )
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.show_comment:
            print(
                "[stop ][{}] {}".format(
                    self.comment, time.strftime("%m%d-%H:%M:%S", time.localtime())
                ).ljust(75, "-")
            )
            print("\t{:.3f} s used".format(time.time() - self.time))
        # 释放content
        self.context = None

    def __str__(self):
        return """Block
\tcomment     : {}
\tcontext_key : {}
""".format(
            self.comment, ",".join(self.context.keys())
        )

    # 对类及其实例未定义的属性有效
    # 若name 不存在于 self.__dir__中,则调用__getattr__
    def __getattr__(self, name):
        cls = type(self)
        res = self.context.setdefault(name, None)
        if res is None:
            raise AttributeError(
                "{.__name__!r} object has no attribute {!r}".format(cls, name)
            )

        return res


def rm_rf(p):
    if not p.exists():
        return

    if p.is_file():
        p.unlink()

    if p.is_dir():
        for i in p.iterdir():
            if i.is_file():
                i.unlink()
            if i.is_dir():
                rm_rf(i)  # 递归
        p.rmdir()


def mv(p_source, p_target):
    p_source = Path(p_source)
    p_target = Path(p_target)
    if p_source.exists() and p_target.parent.exists():
        p_source.replace(p_target)


def str_step_insert(s, step, insert="\n"):
    """
    于s的每step个字符后插入一个insert
    在为图添加超长的文字时
    使用str_step_insert('0123456789012345678901234567',5,insert='\n')
    使字符串换行
    """
    return insert.join(
        [
            s[i * step : (i + 1) * step]
            for i in range(len(s) // step + (0 if len(s) % step == 0 else 1))
        ]
    )


def archive_gzip(
    p_source, decompress=False, out_dir=None, remove_source=True, show=True
):
    """
    gzip 的压缩和解压缩
    """

    def archive_gzip_default(p_source, decompress, out_dir, remove_source, show):
        import gzip
        from shutil import copyfileobj

        func_open_source, func_open_target = open, gzip.open
        if decompress:
            func_open_source, func_open_target = func_open_target, func_open_source

        with func_open_source(p_source, "rb") as f_source:
            with func_open_target(p_target, "wb") as f_target:
                copyfileobj(f_source, f_target)

    def archive_gzip_Linux(p_source, decompress, out_dir, remove_source, show):
        import os

        os.system(
            "gzip -{}c {} > {}".format("d" if decompress else "", p_source, p_target)
        )

    import platform

    handel_func = {"Linux": archive_gzip_Linux}.setdefault(
        platform.system(), archive_gzip_default
    )

    p_source = Path(p_source)
    assert p_source.exists(), "[not exitst] {}".format(p_source)

    out_dir = p_source.parent if out_dir is None else Path(out_dir)
    if decompress:
        assert p_source.match(
            "*.gz"
        ), "[Error] p_source must end with .gz when decompress=True"
        p_target = out_dir.joinpath(p_source.name[:-3])
    else:
        if p_source.name.endswith(".gz"):
            print("[archive_gzip] has commpress {}".format(p_source)) if show else None
            return
        p_target = out_dir.joinpath("{}.gz".format(p_source.name))

    handel_func(p_source, decompress, out_dir, remove_source, show)
    (
        print(
            "[archive_gzip][{}compress] {} -> {}".format(
                "de" if decompress else "", p_source.name, p_target.name
            )
        )
        if show
        else None
    )
    p_source.unlink() if remove_source else None


##################################################
# 使用pickle保存和读取数据
##################################################
def pickle_dump(p_input, **kvargs):
    with open(p_input, "wb") as f:
        pickle.dump(kvargs, f)
    print("[save] {} in {}".format(p_input.name, p_input.parent))


def pickle_loads(p_input):
    data = {}
    with open(p_input, "rb") as f:
        data = pickle.load(f, fix_imports=True, encoding="bytes")
    return data


##################################################
# jupyter UI
##################################################


def init_jquery_UI():
    from IPython.display import display_javascript

    display_javascript(
        """
        if (document.getElementsByClassName("user-container-lcc").length == 0) {
            var ele_body = document.getElementsByTagName("body")[0]
            var ele_div = document.createElement("div")
            var ele_scr = document.createElement("script")
            ele_scr.src = "https://code.jquery.com/jquery-3.6.2.min.js"
            ele_div.className = "user-container-lcc"
            ele_div.insertBefore(ele_scr, ele_div.firstChild)
            ele_body.insertBefore(ele_div, ele_body.firstChild)
        }
    """,
        raw=True,
    )
