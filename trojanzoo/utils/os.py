# -*- coding: utf-8 -*-

import os
import urllib.request
import tarfile
import zipfile

from tqdm import tqdm

def download_and_save(url, savename, output=False):
    try:
        with urllib.request.urlopen(url) as fp:
            data = fp.read()
            fid = open(savename, 'w+b')
            fid.write(data)
            if output:
                print('download succeed: ' + url)
            fid.close()
            return True
    except IOError:
        if output:
            print('download failed: ' + url)
        return False

def untar(file_path, target_path, output=True):
    if output:
        print('Untarring file: ', file_path)
    if not os.path.exists(target_path):
        os.makedirs(target_path)
    tar = tarfile.open(file_path)
    names = tar.getnames()
    for name in tqdm(names):
        tar.extract(name, path=target_path)
    tar.close()
    if output:
        print('Untar finished!')
        print()

def unzip(file_path, target_path, output=True):
    if output:
        print('Unzipping file: ', file_path)
    if not os.path.exists(target_path):
        os.makedirs(target_path)
    with zipfile.ZipFile(file_path) as zf:
        zf.extractall(target_path)
    if output:
        print('Unzip finished!')
        print()

def uncompress(file_path, target_path, output=True):
    for _file in file_path:
        if 'zip' in _file:
            unzip(_file, target_path, output=output)
        elif 'tar.gz' in _file:
            untar(_file, target_path, output=output)
        else:
            raise ValueError('Not Compression File path: %s' % _file)
