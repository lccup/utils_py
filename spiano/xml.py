import lxml.etree as ET


def create_Element(tag, text=None, attrs=None):
    ele = ET.Element(tag, attrib=attrs if attrs else {})
    if text:
        ele.text = text
    return ele


def assemble_multiple_elements(root, elements):
    """
    将多个元素组装到一个根元素下
    root elements

    root E1

    or

    root [E1,E2,E3]

    """
    if isinstance(elements, ET.Element):
        root.append(elements)
    else:
        for i in range(len(elements)):
            if isinstance(elements[i], ET.Element):
                root.append(elements[i])
            else:
                assert i - 1 >= 0 and isinstance(
                    elements[i - 1], ET.Element
                ), "can not append elements[{}] to elemts[{}]\n\telements[{}] = {}".format(
                    i, i - 1, i, elements[i]
                )
                assemble_multiple_elements(elements[i - 1], elements[i])
    return root


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
