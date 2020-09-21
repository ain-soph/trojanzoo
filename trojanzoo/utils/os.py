# -*- coding: utf-8 -*-

import os
import tarfile
import zipfile
from typing import List

from tqdm import tqdm

from trojanzoo.utils.output import ansi

from trojanzoo.utils.config import Config
env = Config.env


def untar(file_path, target_path):
    if not os.path.exists(target_path):
        os.makedirs(target_path)
    tar = tarfile.open(file_path)
    names = tar.getnames()
    if env['tqdm']:
        names = tqdm(names)
    for name in names:
        tar.extract(name, path=target_path)
    if env['tqdm']:
        print('{upline}{clear_line}'.format(**ansi), end='')
    tar.close()


def unzip(file_path, target_path):
    with zipfile.ZipFile(file_path) as zf:
        zf.extractall(target_path)


def uncompress(file_path: List[str], target_path: str, verbose=True):
    if isinstance(file_path, str):
        file_path = [file_path]
    if not os.path.exists(target_path):
        os.makedirs(target_path)
    for _file in file_path:
        if verbose:
            print('Uncompress file: ', _file)
        ext = os.path.splitext(_file)[1]
        if ext in['.zip']:
            unzip(_file, target_path)
        elif ext in ['.tar', '.gz']:
            untar(_file, target_path)
        else:
            raise TypeError('Not Compression File path: %s' % _file)
        if verbose:
            print('Uncompress finished at: ', target_path)
            print()
