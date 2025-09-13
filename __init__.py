from utils_py.general import *
from utils_py import arr
from utils_py import df

pl = None # 绘图
sc = None # 单细胞转录组

def init_pl():
    global pl
    import utils_py.plot as pl

def init_sc():
    global sc
    import utils_py.scanpy as sc


