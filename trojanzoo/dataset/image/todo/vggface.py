# -*- coding: utf-8 -*-
from ..imagefolder import ImageFolder
from package.imports.universal import *
from package.utils.os import uncompress, download_and_save
import torchvision.transforms as transforms

import threading
from tqdm import tqdm


class VGGface(ImageFolder):
    """docstring for dataset"""

    def __init__(self, name='vggface', batch_size=32, n_dim=(224, 224), num_classes=2622, **kwargs):
        super(VGGface, self).__init__(name=name, batch_size=batch_size,
                                      n_dim=n_dim, num_classes=num_classes, **kwargs)

        self.output_par(name='VGGface')
    def initialize(self):
        self.save_idx()
        self.save_img()
        self.split()

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

    def save_idx(self):
        # Download
        file_path = self.download(
            url='https://www.robots.ox.ac.uk/~vgg/data/vgg_face/vgg_face_dataset.tar.gz', file_ext='tar.gz')

        # Untar
        uncompress(file_path=file_path, target_path=self.folder_path)

        # Pack
        print('Moving Index Files...')
        src_folder = self.folder_path+'vgg_face_dataset/files/'
        dst_folder = self.folder_path+'index/'
        if not os.path.exists(dst_folder):
            os.makedirs(dst_folder)
        idx_txt_list = os.listdir(src_folder)
        for txt in idx_txt_list:
            if '.txt' in txt:
                shutil.move(src_folder+txt, dst_folder+txt)
        print('Clean Compression Files...')
        shutil.rmtree(self.folder_path+'vgg_face_dataset')
        print()

    def save_img_for_class(self, file_path, dst_path, thread_num=1000, overlap=False, output=True):
        if not os.path.exists(dst_path):
            os.makedirs(dst_path)
        fid = open(file_path)
        lines = fid.readlines()
        if output:
            lines = tqdm(lines)
        for line in lines:
            line_split = line.split(' ')
            image_id = line_split[0]
            image_url = line_split[1]
            savefile = dst_path + image_id + '.jpg'
            if os.path.exists(savefile) and not overlap:
                continue
            while True:
                if(len(threading.enumerate()) < thread_num):
                    break
            t = threading.Thread(target=download_and_save,
                                 args=(image_url, savefile,))
            t.start()

    def save_img(self, output=True):
        idx_folder = self.folder_path+'index/'
        idx_txt_list = os.listdir(idx_folder)
        counter = 0
        length = len(idx_txt_list)
        for txt in idx_txt_list:
            class_name = txt[:-4]
            if output:
                counter += 1
                print('[%d/%d] ' % (counter, length), class_name)
            self.save_img_for_class(
                idx_folder+txt, self.folder_path+self.name+'/total/'+class_name+'/', output=output)
