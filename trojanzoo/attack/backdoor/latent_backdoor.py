from .badnet import BadNet

from trojanzoo.optim import PGD
from trojanzoo.utils import to_tensor
from trojanzoo.utils.model import AverageMeter

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
# from collections.abc import Callable

class Latent_Backdoor(BadNet):
    r"""
    Latent Backdoor Attack is described in detail in the paper `Latent Backdoor`_ by Yuanshun Yao: 
    
    The authors didn't release source code.

    Args:
        yt_num (int):

    .. _Latent Backdoor:
        http://people.cs.uchicago.edu/~huiyingli/publication/fr292-yaoA.pdf
    """
    name = 'latent_backdoor'

    def __init__(self, preprocess_layer: str = 'features', epsilon: int = 16.0 / 255,
                 poison_num: int = 100, poison_iteration: int = 5000, poison_lr: float = 0.01,
                 lr_decay: bool = False, decay_iteration: int = 2000, decay_ratio: float = 0.95,
                 yt_num: int = 50, ynt_ratio: float = 0.5, mark_area_ratio = 0.04,
                 fine_tune_set_ratio: float = 0.2,
                 **kwargs):
        super().__init__(**kwargs)

        self.param_list['latent_backdoor'] = ['preprocess_layer', 'epsilon',
                                             'poison_num', 'poison_iteration', 'poison_lr',
                                             'decay', 'decay_iteration', 'decay_ratio']

        self.preprocess_layer: str = preprocess_layer   # not use
        self.epsilon: float = epsilon                   # not use

        self.poison_num: int = poison_num
        self.poison_iteration: int = poison_iteration
        self.poison_lr: float = poison_lr

        self.lr_decay: bool = lr_decay                  # not use
        self.decay_iteration: int = decay_iteration     # not use
        self.decay_ratio: float = decay_ratio           # not use

        self.yt_num: int = yt_num
        self.ynt_ratio: float = ynt_ratio
        self.mark_area_ratio: float = mark_area_ratio

        self.fine_tune_set_ratio = fine_tune_set_ratio
        
    # inject backdoor into model after trigger generation
    def attack(self, optimizer: torch.optim.Optimizer, lr_scheduler: torch.optim.lr_scheduler._LRScheduler, iteration: int = None, **kwargs): 
        """
        Given a trained clean model and an generated universal trigger, this function 
        retrains model with a poisoned trainset by following steps:

            step1: use picked ynt images in 'generate_trigger', add trigger on them.

            step2: combine triggered ynt images + picked yt images to a new trainset.

            step3: retrain model with bi-level loss definition (formula (4) in org paper).
        """
        self.generate_trigger()

        # some new self.xxx are created in 'generate_trigger' func.
        ynt_imgs, ynt_labels = self.get_data(self.ynt_sub_loader)  
        ynt_imgs = ynt_imgs * (1-self.mask) + self.mask * self.mark
        poison_set = torch.utils.data.TensorDataset(ynt_imgs.to('cpu'), ynt_labels)

        # add more images from yt label.
        yt_pick_num = int(len(self.yt_loader)*self.yt_ratio)
        yt_train_inds = np.random.choice(list(range(len(self.yt_loader))), yt_pick_num, replace=False)
        yt_train_loader = torch.utils.data.Subset(self.yt_loader, yt_train_inds)
        yt_imgs, yt_labels = self.get_data(yt_sub_loader)
        yt_set = torch.utils.data.TensorDataset(yt_imgs, yt_labels)

        # final trainset = backdoored ynt imgs + many yt imgs
        train_set = torch.utils.data.ConcatDataset((poison_set, yt_set))
        train_loader = self.dataset.get_dataloader(mode=None, dataset=train_set)

        # build validate set, same combination like trainset
        # yt_val_inds = list(set(range(len(self.yt_loader))) - set(yt_train_inds))
        # ynt_val_inds =  list(list(range(len(self.ynt_loader))) - set(yt_train_inds))
        # yt_val_loader = torch.utils.data.Subset(self.yt_loader, yt_train_inds)
        
        val_loader = None
        self.model._train(optimizer=optimizer, lr_scheduler=lr_scheduler,
                          loader_train=train_loader, loader_valid=val_loader,
                          validate_func=self.validate_func, **kwargs)

        self.student_fine_tuning()
        #---------------------------------------------------------------------------#
        """
        Until the end of 'attack', we got a Threat Model that :

            - previous layers are injected backdoor.
            - last layer is replaced by clean.
        """
        #---------------------------------------------------------------------------#


    def generate_trigger(self):
        r"""
        STEP-1: Before trigger generation, we first divide dataset.

        We divide a dataset into two parts: with prefix "yt" and "ynt".

        "yt" only contains images from target class, which always keep clean for trigger 
        generation and injecting backdoor (backdoor training).

        "ynt" contains all other images not in target class, which are used for adding
        trigger and generate trigger, then injecting backdoor.
        """
        print('divide dataset')
        yt = self.target_class
        ynt = list(range(self.dataset.num_classes))
        ynt.pop(yt)
        self.yt_loader = self.dataset.get_dataloader('train', full=True, classes=yt,
                                                     shuffle=True, num_workers=0, drop_last=True)
        self.ynt_loader = self.dataset.get_dataloader('train', full=True, classes=ynt,
                                                      shuffle=True, num_workers=0, drop_last=True)

        """
        STEP-2: Generate Trigger

        We first select some images from yt and ynt, where yt's feature maps are used as the
        groundtruth, and ynt images are used to add trigger --> triggered ynt's feature map 
        as close as yt's --> back propagate to update this trigger.

        Note the universal trigger is the only thing we need to learn, we don't update images.
        """
        img_shape = self.yt_loader[0][0].shape
        height = img_shape[0]
        width = img_shape[1]

        mark_height = int(height * np.sqrt(self.mark_area_ratio))
        mark_width = int(width * np.sqrt(self.mark_area_ratio))

        # we only have 1 mark (trigger) and 1 mask
        self.mark = torch.rand(img_shape)
        self.mark.requires_grad = True
        self.mask = torch.zeros(img_shape)
    
        # trigger locates in right bottom corner, change corresponding mask values
        self.mask[height-mark_height:][:, width-mark_width:] = 1

        # randomly pickup some yt images as feature map groundtruth
        yt_sub_inds = np.random.choice(list(range(len(self.yt_loader))), int(self.poison_num), replace=False)
        self.yt_sub_loader = torch.utils.data.Subset(self.yt_loader, yt_sub_inds)

        # get average feature map from yt's subset images
        _yt_inputs, _ = self.get_data(self.yt_sub_loader)
        yt_featmaps = self.model.get_fm_before_outlayer(_yt_inputs)   # on cpu
        assert yt_featmaps.shape[0]==len(yt_sub_inds), \
               "yt image num mismatch, check 'latent_backdoor' attack-->'generate_trigger' function"
        yt_avg_featmap = yt_featmaps.sum(0) / len(yt_sub_inds)

        # randomly pickup some ynt images for adding trigger
        ynt_pick_num = int(len(self.ynt_loader)*self.ynt_ratio)
        ynt_sub_inds = np.random.choice(list(range(len(self.ynt_loader))), ynt_pick_num, replace=False)
        self.ynt_sub_loader = torch.utils.data.Subset(self.ynt_loader, ynt_sub_inds)


        #------------------------- below is trigger generation process -------------------------#
        self.model.cuda()
        self.mark = self.mark.cuda()
        self.mask = self.mask.cuda()
        yt_avg_featmap = yt_avg_featmap.cuda()

        if not self.poison_lr:
            self.poison_lr = 0.01

        optimizer = optim.Adam([self.mark], lr=self.poison_lr)
        criterion = nn.MSELoss()
        l2_loss = 0.0

        for epoch in range(self.poison_iteration):
            optimizer.zero_grad()
            self.mark = torch.sigmoid(self.mark) # normalize trigger into [0, 1]
            for ynt_ind in ynt_sub_inds:
                ynt_img = self.ynt_loader[ynt_ind][0].cuda()
                ynt_img = ynt_img * (1-self.mask) + self.mask * self.mark
                l2_loss += criterion(ynt_img, yt_avg_featmap, reduction='mean')

            l2_loss = torch.div(l2_loss, len(ynt_sub_inds))
            l2_loss.backward()
            optimizer.step()

        self.model.cpu()
        self.mark = self.mark.cpu()
        self.mask = self.mask.cpu()

        # now we generate an universal trigger (self.mark)


    def student_fine_tuning(self):
        """
        Assume now a victim get the backdoored model, she remove the org classifier 
        with a new one, then only train this classifier.

        The fine-tuned classifier is last (or last two) layer(s) of model. We only
        fine-tune last layer by default.
        """
        train_params = self.model.add_new_last_layer()
        for param in train_params:
            param.requires_grad = True

        fine_tune_loader = self.dataset.get_dataloader('train', full=True,
                              shuffle=True, num_workers=0, drop_last=True)
        fine_tune_inds = np.random.choice(list(range(len(fine_tune_loader))), 
                         int(len(fine_tune_loader)*self.fine_tune_set_ratio), 
                         replace=False)

        # only update last layer parameters
        optimizer = optim.Adam(train_params, lr=self.poison_lr)
        criterion = nn.CrossEntropyLoss()

        self.model.cuda()
        loss, n_sample = 0.0, 0
        for epoch in range(20):
            optimizer.zero_grad()
            for input, label in fine_tune_loader[fine_tune_inds]:
                logits = self.model(input.cuda())
                logits.unsqueeze(0)
                label.unsqueeze(0)
                loss += criterion(logits, label.cuda())
                n_sample += 1
            loss = torch.div(loss, n_sample)
            loss.backward()
            optimizer.step()

        self.model.cpu()


    def evaluation(self):
        pass





