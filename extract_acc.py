# -*- coding: utf-8 -*-

import argparse
from typing import List


def extract_acc(path: str) -> (List[float], List[float]):
    '''
    extract clean and attack accuracies from result files
    '''
    clean_acc, attack_acc = [], []
    lines = open(path, 'r').readlines()
    for l in lines:
        if 'Validate' in l:
            tmp = l.strip().split(',')
            for item in tmp:
                if 'Top1 Acc' in item:
                    acc = item.strip().split(':')[1].strip()
                    acc = float(acc)

                    if 'Validate Clean' in l:
                        clean_acc.append(acc)
                    elif 'Validate Trigger Tgt' in l:
                        attack_acc.append(acc)
    return clean_acc, attack_acc


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', dest='dataset', default='cifar10')
    parser.add_argument('-m', '--model', dest='model', default='resnetcomp18')
    parser.add_argument('--module', dest='module', default='badnet')
    parser.add_argument('--name', dest='name', default='size1')
    args = parser.parse_args()
    path = f'/home/rbp5354/trojanzoo/result/{args.dataset}/{args.model}/{args.module}/{args.name}.txt'
    clean_acc, attack_acc = extract_acc(path)
    print('Clean  Acc: \n', clean_acc)
    print('Poison Acc: \n', attack_acc)
