#!/usr/bin/env python3

# https://github.com/facebookresearch/LaMCTS

from .darts.genotypes import Genotype

operations = ['sep_conv_3x3',
              'max_pool_3x3',
              'skip_connect',
              'sep_conv_5x5']


def gen_code_from_list(sample: list[int], node_num: int = None) -> list[list[int]]:
    node_num = node_num if node_num is not None else len(sample) // 4
    return [[sample[i * 2 + j + (0 if j <= 1 else (node_num - 1) * 2)]
             for j in range(4)] for i in range(node_num)]


def translator(code: list[list[int]], max_node: int = None) -> Genotype:
    # input: code type
    # output: geno type
    max_node = max_node if max_node is not None else len(code)
    normal: list[tuple[str, int]] = []
    reduce: list[tuple[str, int]] = []
    normal_concat: list[int] = list(range(max_node + 2))
    reduce_concat: list[int] = list(range(max_node + 2))
    for cell, concat in zip([normal, reduce], [normal_concat, reduce_concat]):
        for block in range(len(code)):
            cell.append((operations[code[block][0]], code[block][2]))
            cell.append((operations[code[block][1]], code[block][3]))
            if code[block][2] in concat:
                concat.remove(code[block][2])
            if code[block][3] in concat:
                concat.remove(code[block][3])
    if 0 in reduce_concat:
        reduce_concat.remove(0)
    if 1 in reduce_concat:
        reduce_concat.remove(1)
    return Genotype(normal=normal, normal_concat=normal_concat, reduce=reduce, reduce_concat=reduce_concat)
