#!/usr/bin/env python3

from trojanvision.datasets.imagefolder import ImageFolder
from trojanzoo.utils.module import Module


class VGGface2(ImageFolder):
    r"""VGGface2 dataset introduced by Qiong Cao in 2018.
    It inherits :class:`trojanvision.datasets.ImageFolder`.

    See Also:
        * paper: `VGGFace2\: A dataset for recognising faces across pose and age`_
        * website (expired dead link): https://www.robots.ox.ac.uk/~vgg/data/vgg_face2

    Attributes:
        name (str): ``'vggface2'``
        num_classes (int): ``8631`` (Why the papaer claims to have 500 more?)
        data_shape (list[int]): ``[3, 224, 224]``
        valid_set (bool): ``False``

    .. _VGGFace2\: A dataset for recognising faces across pose and age:
        https://arxiv.org/abs/1710.08092
    """
    name: str = 'vggface2'
    num_classes = 8631
    valid_set: bool = False
    # https://docs.qingcloud.com/product/ai/deeplearning/
    url = {'train': 'https://appcenter-deeplearning.sh1a.qingstor.com/dataset/vggface2/vggface2_train.tar.gz',
           'test': 'https://appcenter-deeplearning.sh1a.qingstor.com/dataset/vggface2/vggface2_test.tar.gz'}
    md5 = {'train': '88813c6b15de58afc8fa75ea83361d7f',
           'test': 'bb7a323824d1004e14e00c23974facd3'}
    org_folder_name = {'train': 'train'}


class Sample_VGGface2(VGGface2):

    name: str = 'sample_vggface2'
    num_classes = 20
    url = {}
    org_folder_name = {}

    def initialize_folder(self):
        _dict = Module(self.__dict__)
        _dict.__delattr__('folder_path')
        vggface2 = VGGface2(**_dict)
        vggface2.sample(child_name=self.name, sample_num=self.num_classes)
