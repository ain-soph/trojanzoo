#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from torchvision import get_image_backend
from torchvision.datasets import VisionDataset, DatasetFolder
from torchvision.datasets.folder import has_file_allowed_extension, IMG_EXTENSIONS
import numpy as np
import PIL.Image as Image
import io
import os
import struct
import zipfile
# TODO: Bug of python 3.9.1, collections.abc.Callable[[str],int] regard the [str] as a tuple
from typing import Callable
from typing import Any, Optional, Union, cast


__all__ = ['MemoryDataset']


class MemoryDataset(VisionDataset):
    def __init__(self, data: np.ndarray = None, targets: list[int] = None,
                 root: str = None, **kwargs):
        super().__init__(root=root, **kwargs)
        self.data = data
        self.targets = targets
        if data is None and os.path.isfile(root) and root.endswith('.npz'):
            _dict = np.load(root)
            self.data = _dict['data']
            self.targets = list(_dict['targets'])
        assert len(self.data) == len(self.targets)

    def __getitem__(self, index: Union[int, slice]) -> tuple[Any, Any]:
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]
        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target

    def __len__(self):
        return len(self.targets)


# https://github.com/koenvandesande/vision/blob/read_zipped_data/torchvision/datasets/zippedfolder.py
class ZipFolder(DatasetFolder):
    def __init__(self, root: str, transform: Optional[Callable] = None, target_transform: Optional[Callable] = None,
                 is_valid_file: Optional[Callable[[str], bool]] = None) -> None:
        if not root.endswith('.zip'):
            raise TypeError("Need to ZIP file for data source: ", root)
        self.root_zip = ZipLookup(os.path.realpath(root))
        super().__init__(root, self.zip_loader, IMG_EXTENSIONS if is_valid_file is None else None,
                         transform=transform, target_transform=target_transform, is_valid_file=is_valid_file)
        self.imgs = self.samples

    def make_dataset(
        self,
        directory: str,
        class_to_idx: dict[str, int],
        extensions: Optional[tuple[str, ...]] = None,
        is_valid_file: Optional[Callable[[str], bool]] = None,
    ) -> list[tuple[str, int]]:
        instances = []
        both_none = extensions is None and is_valid_file is None
        both_something = extensions is not None and is_valid_file is not None
        if both_none or both_something:
            raise ValueError("Both extensions and is_valid_file cannot be None or not None at the same time")
        if extensions is not None:
            def is_valid_file(x: str) -> bool:
                return has_file_allowed_extension(x, cast(tuple[str, ...], extensions))
        is_valid_file = cast(Callable[[str], bool], is_valid_file)
        for filepath in self.root_zip.keys():
            if is_valid_file(filepath):
                _, target_class = os.path.split(os.path.dirname(filepath))
                item = filepath, class_to_idx[target_class]
                instances.append(item)
        return instances

    def zip_loader(self, path) -> Any:
        f = self.root_zip[path]
        if get_image_backend() == 'accimage':
            try:
                import accimage  # type: ignore
                return accimage.Image(f)
            except IOError:
                pass   # fall through to PIL
        return Image.open(f).convert('RGB')

    def _find_classes(self, *args, **kwargs):
        """
        Finds the class folders in a dataset.
        Args:
            dir (string): Root directory path.
        Returns:
            tuple: (classes, class_to_idx) where classes are relative to (dir), and class_to_idx is a dictionary.
        Ensures:
            No class is a subdirectory of another.
        """
        classes = set()
        for filepath in self.root_zip.keys():
            root, target_class = os.path.split(os.path.dirname(filepath))
            if root:
                classes.add(target_class)
        classes = list(classes)
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx


class ZipLookup(object):
    def __init__(self, filename):
        self.root_zip_filename = filename
        self.root_zip_lookup: dict[str, tuple[int, int]] = {}

        with zipfile.ZipFile(filename, "r") as root_zip:
            for info in root_zip.infolist():
                if info.filename[-1] == '/':
                    # skip directories
                    continue
                if info.compress_type != zipfile.ZIP_STORED:
                    raise ValueError("Only uncompressed ZIP file supported: " + info.filename)
                if info.compress_size != info.file_size:
                    raise ValueError("Must be the same when uncompressed")
                self.root_zip_lookup[info.filename] = (info.header_offset, info.compress_size)

    def __getitem__(self, path):
        z = open(self.root_zip_filename, "rb")
        header_offset, size = self.root_zip_lookup[path]

        z.seek(header_offset)
        fheader = z.read(zipfile.sizeFileHeader)
        fheader = struct.unpack(zipfile.structFileHeader, fheader)
        offset = header_offset + zipfile.sizeFileHeader + fheader[zipfile._FH_FILENAME_LENGTH] + \
            fheader[zipfile._FH_EXTRA_FIELD_LENGTH]

        z.seek(offset)
        f = io.BytesIO(z.read(size))
        f.name = path
        z.close()
        return f

    def keys(self):
        return self.root_zip_lookup.keys()
