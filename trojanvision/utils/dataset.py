#!/usr/bin/env python3

from torchvision import get_image_backend
from torchvision.datasets import VisionDataset, DatasetFolder
from torchvision.datasets.folder import has_file_allowed_extension, IMG_EXTENSIONS
import numpy as np
import PIL.Image as Image
import io
import os
import tarfile
import zipfile
from typing import Any, cast
from collections.abc import Callable


__all__ = ['MemoryDataset', 'ZipFolder']


# TODO: Need reorganization
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

    def __getitem__(self, index: int | slice) -> tuple[Any, Any]:
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
    def __init__(self, root: str, transform: None | Callable = None, target_transform: None | Callable = None,
                 is_valid_file: None | Callable[[str], bool] = None, memory: bool = True) -> None:
        if not root.lower().endswith('.zip'):
            raise TypeError("Need zip file for data source: ", root)
        if memory:
            with open(root, 'rb') as zf:
                data = zf.read()
            self.root_data = zipfile.ZipFile(io.BytesIO(data), 'r')
        else:
            self.root_data = zipfile.ZipFile(root, 'r')
        super().__init__(root, self.zip_loader, IMG_EXTENSIONS if is_valid_file is None else None,
                         transform=transform, target_transform=target_transform, is_valid_file=is_valid_file)
        self.imgs = self.samples

    @staticmethod
    def initialize_from_folder(root: str, zip_path: str = None):
        root = os.path.normpath(root)
        folder_dir, folder_base = os.path.split(root)
        if zip_path is None:
            zip_path = os.path.join(folder_dir, f'{folder_base}_store.zip')
        with zipfile.ZipFile(zip_path, mode='w', compression=zipfile.ZIP_STORED) as zf:
            for walk_root, walk_dirs, walk_files in os.walk(root):
                zip_root = walk_root.removeprefix(folder_dir)
                for _file in walk_files:
                    org_path = os.path.join(walk_root, _file)
                    dst_path = os.path.join(zip_root, _file)
                    zf.write(org_path, dst_path)

    def make_dataset(
        self,
        directory: str,
        class_to_idx: dict[str, int],
        extensions: None | tuple[str, ...] = None,
        is_valid_file: None | Callable[[str], bool] = None,
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
        for filepath in self.root_data.namelist():
            if is_valid_file(filepath):
                target_class = os.path.basename(os.path.dirname(filepath))
                instances.append((filepath, class_to_idx[target_class]))
        return instances

    def zip_loader(self, path: str) -> Image.Image:
        f = self.root_data.open(path, 'r')
        if get_image_backend() == 'accimage':
            try:
                import accimage  # type: ignore
                return accimage.Image(f)
            except IOError:
                pass   # fall through to PIL
        return Image.open(f).convert('RGB')

    def _find_classes(self, *args, **kwargs) -> tuple[list[str], dict[str, int]]:
        r"""Finds the class folders in a dataset.

        Args:
            dir (str): Root directory path.

        Returns:
            tuple: (classes, class_to_idx) where classes are relative to (dir), and class_to_idx is a dictionary.

        Ensures:
            No class is a subdirectory of another.
        """
        class_set = set()
        for filepath in self.root_data.namelist():
            root, target_class = os.path.split(os.path.dirname(filepath))
            if root:
                class_set.add(target_class)
        classes = list(class_set)
        classes.sort()  # TODO: Pylance issue
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx


class TarFolder(DatasetFolder):
    def __init__(self, root: str, transform: None | Callable = None, target_transform: None | Callable = None,
                 is_valid_file: None | Callable[[str], bool] = None, memory: bool = True) -> None:
        if not root.lower().endswith('.tar'):
            raise TypeError("Need tar file for data source: ", root)
        if memory:
            with open(root, 'rb') as zf:
                data = zf.read()
            self.root_data = tarfile.open(io.BytesIO(data), 'r')
        else:
            self.root_data = tarfile.open(root, 'r')
        super().__init__(root, self.tar_loader, IMG_EXTENSIONS if is_valid_file is None else None,
                         transform=transform, target_transform=target_transform, is_valid_file=is_valid_file)
        self.imgs = self.samples

    @staticmethod
    def initialize_from_folder(root: str, tar_path: str = None) -> None:
        root = os.path.normpath(root)
        folder_dir, folder_base = os.path.split(root)
        if tar_path is None:
            tar_path = os.path.join(folder_dir, f'{folder_base}.tar')
        with tarfile.open(tar_path, mode='w') as zf:
            for walk_root, walk_dirs, walk_files in os.walk(root):
                tar_root = walk_root.removeprefix(folder_dir)
                for _file in walk_files:
                    org_path = os.path.join(walk_root, _file)
                    dst_path = os.path.join(tar_root, _file)
                    zf.add(org_path, dst_path)

    def make_dataset(
        self,
        directory: str,
        class_to_idx: dict[str, int],
        extensions: None | tuple[str, ...] = None,
        is_valid_file: None | Callable[[str], bool] = None,
    ) -> list[tuple[str, int]]:
        instances: list[tuple[str, int]] = []
        both_none = extensions is None and is_valid_file is None
        both_something = extensions is not None and is_valid_file is not None
        if both_none or both_something:
            raise ValueError("Both extensions and is_valid_file cannot be None or not None at the same time")
        if extensions is not None:
            def is_valid_file(x: str) -> bool:
                return has_file_allowed_extension(x, cast(tuple[str, ...], extensions))
        is_valid_file = cast(Callable[[str], bool], is_valid_file)
        for filepath in self.root_data.getnames():
            if is_valid_file(filepath):
                target_class = os.path.basename(os.path.dirname(filepath))
                instances.append((filepath, class_to_idx[target_class]))
        return instances

    def tar_loader(self, path: str) -> Image.Image:
        f = self.root_data.extract(path)
        if get_image_backend() == 'accimage':
            try:
                import accimage  # type: ignore
                return accimage.Image(f)
            except IOError:
                pass   # fall through to PIL
        return Image.open(f).convert('RGB')

    def _find_classes(self, *args, **kwargs) -> tuple[list[str], dict[str, int]]:
        """
        Finds the class folders in a dataset.
        Args:
            dir (str): Root directory path.
        Returns:
            tuple: (classes, class_to_idx) where classes are relative to (dir), and class_to_idx is a dictionary.
        Ensures:
            No class is a subdirectory of another.
        """
        classes: set[str] = set()
        for filepath in self.root_data.getnames():
            root, target_class = os.path.split(os.path.dirname(filepath))
            if root:
                classes.add(target_class)
        classes = list(classes)
        classes.sort()  # TODO: Pylance issue
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx
