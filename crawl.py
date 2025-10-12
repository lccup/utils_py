"""
# crawl

爬虫相关函数

核心包:requests,lxml

> variable
    session 单例
    headers_default

> function
    get_update_headers
    get
    load_etree
"""


from pathlib import Path
import json
import requests
from lxml import etree

session = requests.session()

headers_default = {'User-Agent':
'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36 Edg/125.0.0.0'
}

# ----------------------------------------
# requests发送网络请求
# ----------------------------------------


def get_update_headers(dict_update={},headers = headers_default):
    """
    使用dict_update对headers进行更新
    首先对headers进行copy, 再update, 不会对外部的headers进行修改

    Parameters
    ----------
    dict_update : dict (default: {})

    headers : dict (default: ut.crawl.headers_default)
    """
    headers = headers.copy()
    headers.update(dict_update)
    return headers

def get(url,headers = None,data = {},
        session=session,
        status_limit = True,
        status_allow=[200],
        *args,**kvargs):
    """
    由session调用get

    Parameters
    ----------
    headers : dict (default: None)
        若为None,则get_update_headers()为其赋值
        该函数默认返回ut.crawl.headers_default
    """
    if headers is None:
        headers = get_update_headers()
    r = session.get(url=url,headers=headers,data=data,*args,**kvargs)
    if status_limit:
        assert r.status_code in status_allow,'[status is not allowed] {}'.format(r.status_code)
    return r

# ----------------------------------------
# 使用lxml.etree进行xpath解析
# ----------------------------------------

def load_etree(str_or_path):
    txt = None
    if isinstance(str_or_path, str):
        try:
            str_or_path = (
                Path(str_or_path) if Path(str_or_path).exists() else str_or_path
            )
        except Exception as e:
            txt = str_or_path
    if isinstance(str_or_path, Path):
        txt = Path(str_or_path).read_text()
    assert not txt is None
    return etree.HTML(txt)

def xpath_set_default(ele, xpath, default=None):
    """
    类似dict.setdefault
    由于xpath 的结果为list,当元素仅可能为1时需要[0]取值
    而未查找到时需额外判断

    故统一使用该函数处理,
    若xpath为空,返回default,否则返回查找到的第一个元素
    """
    res = ele.xpath(xpath)
    return res[0] if len(res) > 0 else default

