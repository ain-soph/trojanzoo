# -*- coding: utf-8 -*-

import os
import urllib.request
import tarfile
import zipfile
from typing import List

from tqdm import tqdm


def download_and_save(url, savename, verbose=False):
    try:
        with urllib.request.urlopen(url) as fp:
            data = fp.read()
            fid = open(savename, 'w+b')
            fid.write(data)
            if verbose:
                print('download succeed: ' + url)
            fid.close()
            return True
    except IOError:
        if verbose:
            print('download failed: ' + url)
        return False


def untar(file_path, target_path):
    if not os.path.exists(target_path):
        os.makedirs(target_path)
    tar = tarfile.open(file_path)
    names = tar.getnames()
    for name in tqdm(names):
        tar.extract(name, path=target_path)
    tar.close()


def unzip(file_path, target_path):
    with zipfile.ZipFile(file_path) as zf:
        zf.extractall(target_path)


def uncompress(file_path: List[str], target_path: str, verbose=False):
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
