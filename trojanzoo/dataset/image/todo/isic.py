# -*- coding: utf-8 -*-
from ..imagefolder import ImageFolder
from package.imports.universal import *
import torchvision.transforms as transforms

import pandas as pd
from tqdm import tqdm


class ISIC(ImageFolder):
    """docstring for dataset"""

    def __init__(self, name='isic_abstract', batch_size=32, n_dim=(224, 224), **kwargs):
        super(ISIC, self).__init__(
            name=name, batch_size=batch_size, n_dim=n_dim, **kwargs)
        self.url = {}
        self.org_folder_name = {}

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

    def split_class(self):
        print('Collect Label Information from CSV file: ',
              self.folder_path+'label.csv')
        obj = pd.read_csv(self.folder_path+'label.csv')
        labels = list(obj.columns.values)
        org_dict = {}
        for label in labels:
            org_dict[label] = np.array(obj[label]) if label == 'image' else np.array([
                bool(value) for value in obj[label]])
        new_dict = {}
        for label in labels[1:]:
            new_dict[label] = org_dict['image'][org_dict[label]]

        print('Splitting dataset to class folders ...')

        src_folder = self.folder_path+self.name+'/total/'
        for label in tqdm(labels[1:]):
            seq = new_dict[label]

            dst_folder = self.folder_path+self.name+'/total/'+label+'/'

            if not os.path.exists(dst_folder):
                os.makedirs(dst_folder)
            for img in seq:
                src = src_folder+img+'.jpg'
                dest = dst_folder+img+'.jpg'
                shutil.move(src, dest)

    def split(self, *args, **kwargs):
        self.split_class()
        super(ISIC, self).split(*args, **kwargs)


class ISIC2018(ISIC):
    """docstring for dataset"""

    def __init__(self, name='isic2018', num_classes=7, default_model='resnet101', **kwargs):
        super(ISIC2018, self).__init__(
            name=name, num_classes=num_classes, default_model=default_model, **kwargs)
        self.url['train'] = 'https://challenge.kitware.com/api/v1/item/5ac20fc456357d4ff856e139/download'
        self.org_folder_name['train'] = 'ISIC2018_Task3_Training_Input'

        self.output_par(name='ISIC2018')


class ISIC2019(ISIC):
    """docstring for dataset"""

    def __init__(self, name='isic2019', num_classes=9, **kwargs):
        super(ISIC2019, self).__init__(
            name=name, num_classes=num_classes, **kwargs)
        self.url['train'] = 'https://s3.amazonaws.com/isic-challenge-2019/ISIC_2019_Training_Input.zip'
        self.org_folder_name['train'] = 'ISIC_2019_Training_Input'

        self.output_par(name='ISIC2019')
