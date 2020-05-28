# -*- coding: utf-8 -*-
from ..imagefolder import ImageFolder
import torchvision.transforms as transforms


class ImageNet(ImageFolder):

    def __init__(self, name='imagenet', n_dim=(224, 224), num_classes=1000,
                 norm_par={'mean': [0.485, 0.456, 0.406],
                           'std': [0.229, 0.224, 0.225], },
                 **kwargs):
        super().__init__(name=name, n_dim=n_dim, num_classes=num_classes,
                         norm_par=norm_par, **kwargs)
        self.url = {}
        self.url['train'] = 'http://www.image-net.org/challenges/LSVRC/2012/nnoupb/ILSVRC2012_img_train.tar'
        self.url['valid'] = 'http://www.image-net.org/challenges/LSVRC/2012/nnoupb/ILSVRC2012_img_val.tar'
        self.url['test'] = 'http://www.image-net.org/challenges/LSVRC/2012/nnoupb/ILSVRC2012_img_test.tar'
        self.org_folder_name = {}

        self.output_par(name='ImageNet')

    def initialize(self):
        pass

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

    def get_full_dataset(self, mode):
        if mode == 'valid' and self.name == 'imagenet':
            mode = 'val'
        return super().get_full_dataset(mode)