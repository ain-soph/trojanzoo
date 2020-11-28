from typing import List
import numpy as np
import re


def split(string: str) -> List[float]:
    string = re.sub(u"\\(.*?\\)|\\{.*?}|\\[.*?]", "", string)
    str_list = string.split('&')
    str_list = [i.strip() for i in str_list]
    float_list = np.array([float(i) for i in str_list])
    return float_list


def split_conf(string: str) -> List[float]:
    pattern = re.compile(r'[(](.*?)[)]', re.S)

    str_list = string.split('&')
    str_list = [i.strip() for i in str_list]
    conf_list = [None] * len(str_list)
    for i in range(len(str_list)):
        sub_str = str_list[i]
        result = re.findall(pattern, sub_str)
        if len(result) == 1:
            conf_list[i] = float(result[0])
    return conf_list


def merge_str(float_list: List[float], conf_list: List[float] = None, sign=True, base_format='.1f', confidence_format='.3f') -> str:
    final_list = []
    if conf_list is None:
        conf_list = [None] * len(float_list)
    for i in range(len(float_list)):
        a = f'{float_list[i]:{base_format}}'
        if sign and float_list[i] > 0 and a != '0.0':
            a = '+' + a
        if a == '-0.0':
            a = '0.0'
        if conf_list[i] is not None:
            a = a + f' ({conf_list[i]:{confidence_format}})'
        final_list.append(a)
    return ' & '.join(final_list)


# base = '99.96 (1.000) & 98.75 (0.994) & 98.40 (0.995)           & 96.00 (0.982)'
# tgt = '10.08 (0.807) & 10.44 (0.601) & 13.42 (0.643) & 10.74 (0.728) & 58.72 (0.889) & 72.38 (0.822) & 0.00 (0.811)  & 10.86 (0.644) & 73.07 (0.826)'

# base_list = split(base)
# tgt_list = split(tgt)
# base_conf_list = split_conf(base)
# tgt_conf_list = split_conf(tgt)
# final_list = tgt_list - base_list

# print(merge_str(base_list, base_conf_list, sign=False))
# print(merge_str(final_list, tgt_conf_list))
order_idx = [0, 1, 4, 2, 5, 6, 3]

base = '0.700  & 0.417    & 0.214  & 0.133 & 0.700      & 0.000        & 0.417'
base_list = split(base)
base_list = [base_list[i] for i in order_idx]
print(merge_str(base_list, sign=False, base_format='.2f'))
