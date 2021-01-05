#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json

chn2eng = {
    '鱼': 'fish',
    '鸟': 'bird',
    '龟': 'turtle',
    '蜥蜴': 'lizard',
    '蛇': 'snake',
    '蜘蛛': 'spider',
    '螃蟹': 'crab',
    '狗': 'dog',
    '狼': 'wolf',
    '猫': 'cat',
    '昆虫': 'insect',
    '鼠': 'rat',
    '猴子': 'monkey',
    '菌菇': 'mushroom',
    '羊': 'sheep',
    '猪': 'pig',
    '牛': 'cow',
    '兔子': 'rabbit',
    '蝴蝶': 'butterfly',
    '狐狸': 'fox',
}
if __name__ == '__main__':

    eng2chn = {}
    for (k, v) in chn2eng.items():
        eng2chn[v] = k
    class_dict = {}
    for k in eng2chn.keys():
        class_dict[k] = []
    with open('./class.txt', encoding='utf-8') as txt:
        lines = txt.readlines()
        class_list = [a.strip().split(' ') for a in lines]
    for pair in class_list:
        print(pair)
        if pair[1] in chn2eng.keys():
            class_dict[chn2eng[pair[1]]].append(pair[0])

    with open('./class_dict.json', 'w') as f:
        json.dump(class_dict, f)
