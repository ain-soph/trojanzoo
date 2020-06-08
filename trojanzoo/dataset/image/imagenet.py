# -*- coding: utf-8 -*-
from ..imagefolder import ImageFolder
import torchvision.transforms as transforms
from torchvision.datasets import ImageNet as ImageNet_Official

from trojanzoo.utils.param import Module
import os
import json
from trojanzoo import __file__ as root_file
root_dir = os.path.dirname(os.path.abspath(root_file))


class ImageNet(ImageFolder):

    name = 'imagenet'
    num_classes = 1000
    n_dim = (224, 224)
    url = {
        'train': 'http://www.image-net.org/challenges/LSVRC/2012/nnoupb/ILSVRC2012_img_train.tar',
        'valid': 'http://www.image-net.org/challenges/LSVRC/2012/nnoupb/ILSVRC2012_img_val.tar',
        'test': 'http://www.image-net.org/challenges/LSVRC/2012/nnoupb/ILSVRC2012_img_test.tar',
    }
    org_folder_name = {}

    def __init__(self, norm_par={'mean': [0.485, 0.456, 0.406],
                                 'std': [0.229, 0.224, 0.225], },
                 **kwargs):
        super().__init__(norm_par=norm_par, **kwargs)

    def initialize(self):
        ImageNet_Official(root=self.folder_path, split='train', download=True)
        ImageNet_Official(root=self.folder_path, split='val', download=True)

    @staticmethod
    def get_transform(mode):
        if mode == 'train':
            transform = transforms.Compose([
                transforms.RandomResizedCrop((224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.CenterCrop((224, 224)),
                transforms.ToTensor(),
            ])
        return transform

    def get_full_dataset(self, mode):
        if mode == 'valid' and self.name == 'imagenet':
            mode = 'val'
        return super().get_full_dataset(mode)


class Sample_ImageNet(ImageNet):

    name = 'sample_imagenet'
    num_classes = 20
    n_dim = (224, 224)
    url = {}
    org_folder_name = {}

    def initialize(self):
        _dict = Module(self.__dict__)
        _dict.__delattr__('folder_path')
        imagenet = ImageNet(**_dict)
        class_dict: dict = {}
        with open(root_dir+'/data/{}/data/class_dict.json'.format(self.name), 'r', encoding='utf-8') as f:
            class_dict: dict = json.load(f)
        imagenet.sample(child_name=self.name, class_dict=class_dict)
