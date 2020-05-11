# -*- coding: utf-8 -*-
from ..imagefolder import ImageFolder
from package.imports.universal import *
import torchvision.transforms as transforms


class ImageNet(ImageFolder):
    """docstring for dataset"""

    def __init__(self, name='imagenet', batch_size=32, n_dim=(224, 224), norm_par={
            'mean': [0.485, 0.456, 0.406],
            'std': [0.229, 0.224, 0.225], }, num_classes=1000, **kwargs):
        super(ImageNet, self).__init__(name=name, batch_size=batch_size, norm_par=norm_par,
                                       n_dim=n_dim, num_classes=num_classes, **kwargs)
        self.url = {}
        self.url['train'] = 'http://www.image-net.org/challenges/LSVRC/2012/nnoupb/ILSVRC2012_img_train.tar'
        self.url['valid'] = 'http://www.image-net.org/challenges/LSVRC/2012/nnoupb/ILSVRC2012_img_val.tar'
        self.url['test'] = 'http://www.image-net.org/challenges/LSVRC/2012/nnoupb/ILSVRC2012_img_test.tar'
        self.org_folder_name = {}

        self.output_par(name='ImageNet')

    def initialize(self):

        pass

    def get_full_dataset(self, mode):
        if mode == 'valid' and self.name == 'imagenet':
            mode = 'val'
        return super(ImageNet, self).get_full_dataset(mode)

    def get_transform(self, mode):
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
