# -*- coding: utf-8 -*-

# CUDA_VISIBLE_DEVICES=1 python saliency_map.py --dataset sample_imagenet  --width 5 --height 5 --verbose --pretrain  --batch_size 5 --attack badnet


from trojanzoo.parser import Parser_Dataset, Parser_Model, Parser_Train, Parser_Seq, Parser_Mark, Parser_Attack

from trojanzoo.dataset import ImageSet
from trojanzoo.model import ImageModel
from trojanzoo.attack import BadNet
from trojanzoo.utils.mark import Watermark
from trojanzoo.utils.tensor import save_tensor_as_img, to_numpy, save_numpy_as_img
import cv2
from trojanzoo.utils.config import Config
env = Config.env
import numpy as np
import warnings
warnings.filterwarnings("ignore")
import torch
from trojanzoo.utils.data import MyDataset

def show_cam_on_image(img, mask, filepath):
    print(img.shape, mask.shape)
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    img = np.transpose(img,(1,2,0))

    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    print("the shape of cam:",cam.shape)
    cv2.imwrite(filepath, np.uint8(255 * cam))

if __name__ == '__main__':
    parser = Parser_Seq(Parser_Dataset(), Parser_Model(), Parser_Train(),
                        Parser_Mark(), Parser_Attack())
    parser.parse_args()
    parser.get_module()

    dataset: ImageSet = parser.module_list['dataset']
    model: ImageModel = parser.module_list['model']
    optimizer, lr_scheduler, train_args = parser.module_list['train']
    mark: Watermark = parser.module_list['mark']
    attack: BadNet = parser.module_list['attack']



    chooseset = dataset.get_dataset(mode='train')
    subset, _ = dataset.split_set(chooseset, length=1)

    clean_loader = dataset.get_dataloader(mode='train', dataset=subset)
    _input, _label = next(iter(torch.utils.data.DataLoader(subset, batch_size=len(subset), num_workers=0)))
    orig_img= np.array(_input[0].cpu())
    save_numpy_as_img(r'./bad_orig_image.png', orig_img)
    
    print("_label:", _label)
    print("predicted label on benign image:", model.get_class(_input.to(env['device'])))

    poison_input = attack.add_mark(_input)
    print("predicted_label on backdoor image:", model.get_class(poison_input.to(env['device'])))

    poison_img= np.array(poison_input[0].cpu())
    save_numpy_as_img(r'./bad_backdoor_image.png', poison_img)

    poison_label = attack.target_class * torch.ones_like(_label)
    newset = MyDataset(poison_input, poison_label)
    backdoor_loader = dataset.get_dataloader(mode='train', dataset=newset)

    backdoor_target_saliency_maps = model.grad_cam(poison_input.to(env['device']), attack.target_class)   # (N, 1, H, W)
    backdoor_orig_saliency_maps = model.grad_cam(poison_input.to(env['device']), _label) 
    benign_target_saliency_maps = model.grad_cam(_input.to(env['device']), attack.target_class)        # (N, 1, H, W)
    benign_orig_saliency_maps = model.grad_cam(_input.to(env['device']), _label)     # (N, 1, H, W)
    show_cam_on_image(poison_img, backdoor_target_saliency_maps, './cleanmodel_backdoor_target.png')
    show_cam_on_image(poison_img, backdoor_orig_saliency_maps, './cleanmodel_backdoor_orig.png')
    show_cam_on_image(orig_img, benign_target_saliency_maps, './cleanmodel_benign_target.png')
    show_cam_on_image(orig_img, benign_orig_saliency_maps, './cleanmodel_benign_orig.png')

    attack.load()
    print("predicted label on benign image:", attack.model.get_class(_input.to(env['device'])))
    print("predicted_label on backdoor image:", attack.model.get_class(poison_input.to(env['device'])))

    backdoor_target_saliency_maps = attack.model.grad_cam(poison_input.to(env['device']), attack.target_class)   # (N, 1, H, W)
    backdoor_orig_saliency_maps = attack.model.grad_cam(poison_input.to(env['device']), _label) 
    benign_target_saliency_maps = attack.model.grad_cam(_input.to(env['device']), attack.target_class)        # (N, 1, H, W)
    benign_orig_saliency_maps = attack.model.grad_cam(_input.to(env['device']), _label)     # (N, 1, H, W)
    show_cam_on_image(poison_img, backdoor_target_saliency_maps, './badmodel_backdoor_target.png')
    show_cam_on_image(poison_img, backdoor_orig_saliency_maps, './badmodel_backdoor_orig.png')
    show_cam_on_image(orig_img, benign_target_saliency_maps, './badmodel_benign_target.png')
    show_cam_on_image(orig_img, benign_orig_saliency_maps, './badmodel_benign_orig.png')



