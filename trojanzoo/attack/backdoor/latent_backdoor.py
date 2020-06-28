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


    .. _Latent Backdoor:
        http://people.cs.uchicago.edu/~huiyingli/publication/fr292-yaoA.pdf
    """
    name = 'latent_backdoor'

    def __init__(self, poison_num: int = 100, poison_iteration: int = 100, poison_lr: float = 0.01,
                 lr_decay: bool = False, decay_iteration: int = 2000, decay_ratio: float = 0.95,
                 ynt_ratio: float = 0.1, mark_area_ratio = 0.09, val_ratio = 0.1, 
                 fine_tune_set_ratio: float = 0.1,
                 **kwargs):
        super().__init__(**kwargs)

        self.param_list['latent_backdoor'] = ['poison_num', 'poison_iteration', 'poison_lr',
                                             'decay', 'decay_iteration', 'decay_ratio',
                                             'ynt_ratio', 'mark_area_ratio', 'val_ratio', 
                                             'fine_tune_set_ratio']

        self.poison_num: int = poison_num
        self.poison_iteration: int = poison_iteration
        self.poison_lr: float = poison_lr

        self.lr_decay: bool = lr_decay                  # not use
        self.decay_iteration: int = decay_iteration     # not use
        self.decay_ratio: float = decay_ratio           # not use

        self.ynt_ratio: float = ynt_ratio
        self.val_ratio: float  = val_ratio
        self.mark_area_ratio: float = mark_area_ratio
        self.fine_tune_set_ratio: float  = fine_tune_set_ratio  
        

    # inject backdoor into model after trigger generation
    def attack(self, optimizer: torch.optim.Optimizer, lr_scheduler: torch.optim.lr_scheduler._LRScheduler, iteration: int = None, **kwargs): 
        """
        Given a trained clean model and an generated universal trigger, this function 
        retrains model with a poisoned trainset by following steps:

            step1: use picked ynt images in 'generate_trigger', add trigger on them.

            step2: combine triggered ynt images + picked yt images to a new trainset.

            step3: retrain model with bi-level loss definition (formula (4) in org paper).

        
        About dataset divisions, we totally have following of them:

        group1: created in 'generate_trigger'
            self.yt_imgs, self.yt_labels: all images in target label (yt)
            self.ynt_imgs, self.ynt_labels: all images not in target label (y\t)
            self.yt_sub_inds: indices of some yt images, used as feature map groundtruth.
            self.ynt_sub_inds: indices of some y\t images, used to add trigger then measure feature map distance.

        group2: created in 'attack', i.e. inject backdoor process
            self.train_loader: trainset with poisoned images.
            self.val_loader: validate set with poisoned images, disjoint with trainset.

        group3: created in 'student_fine_tuning'
            None self.xxx, there is a 'fine_tune_loader' contains randomly picked clean data
            for fine-tuning a clean output layer.
        """
        self.generate_trigger()

        # step1
        # some sub loaders are created in 'generate_trigger'
        ynt_sub_imgs = self.ynt_imgs[self.ynt_sub_inds]
        ynt_sub_labels = self.ynt_labels[self.ynt_sub_inds]
        ynt_sub_imgs = ynt_sub_imgs * (1-self.mask) + self.mask * self.mark
        ynt_set = torch.utils.data.TensorDataset(ynt_sub_imgs.to('cpu'), ynt_sub_labels) # now poisoned

        yt_sub_imgs = self.yt_imgs[self.yt_sub_inds]
        yt_sub_labels = self.yt_labels[self.yt_sub_inds]
        yt_set = torch.utils.data.TensorDataset(yt_sub_imgs, yt_sub_labels)

        poison_set = torch.utils.data.ConcatDataset((ynt_set, yt_set))

        val_inds = np.random.choice(list(range(len(poison_set))), 
                                    int(len(poison_set)*self.val_ratio), 
                                    replace=False)
        train_inds = list(set(range(len(poison_set)))-set(val_inds))
        
        train_set = torch.utils.data.Subset(poison_set, train_inds)
        val_set = torch.utils.data.Subset(poison_set, val_inds)
        
        self.train_loader = self.dataset.get_dataloader(mode=None, dataset=train_set)
        self.val_loader = self.dataset.get_dataloader(mode=None, dataset=val_set)
        
        print("...injecting backdoor into DNN")
        self.model.cuda()
        self.model.train()
        self.model._train(optimizer=optimizer, lr_scheduler=lr_scheduler,
                          loader_train=self.train_loader, loader_valid=self.val_loader,
                          get_data=None, validate_func=None, **kwargs)

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
        print("...dividing dataset")
        yt = self.target_class
        ynt = list(range(self.dataset.num_classes))
        ynt.pop(yt)
        yt_loader = self.dataset.get_dataloader('train', full=True, classes=[yt],
                                                     shuffle=True, num_workers=0, drop_last=True)
        ynt_loader = self.dataset.get_dataloader('train', full=True, classes=ynt,
                                                      shuffle=True, num_workers=0, drop_last=True)
        self.yt_imgs, self.yt_labels  = self.loader_to_dataset(yt_loader)
        self.ynt_imgs, self.ynt_labels = self.loader_to_dataset(ynt_loader)

        yt_num = len(self.yt_labels)
        ynt_num = len(self.ynt_labels)

        # randomly pickup some yt images as feature map groundtruth
        # self.yt_sub_loader: contains yt images that used as feature map groundtruth
        self.yt_sub_inds = np.random.choice(list(range(yt_num)), int(self.poison_num), replace=False)
        
        # randomly pickup some ynt images for adding trigger
        # self.ynt_sub_loader: contains some y\t images that used for adding trigger
        ynt_pick_num = int(ynt_num * self.ynt_ratio)
        self.ynt_sub_inds = np.random.choice(list(range(ynt_num)), ynt_pick_num, replace=False)

        """
        STEP-2: Generate Trigger

        We first select some images from yt and ynt, where yt's feature maps are used as the
        groundtruth, and ynt images are used to add trigger --> triggered ynt's feature map 
        as close as yt's --> back propagate to update this trigger.

        Note the universal trigger is the only thing we need to learn, we don't update images.
        """
        # self.yt_imgs.shape: (BatchSize, Channel, Height, Wdith)
        img_shape = self.yt_imgs.shape[1:]
        height = img_shape[1]
        width = img_shape[2]
        
        mark_height = int(height * np.sqrt(self.mark_area_ratio))
        mark_width = int(width * np.sqrt(self.mark_area_ratio))

        # we only have 1 mark (trigger) and 1 mask
        self.mark = torch.rand(img_shape)
        self.mask = torch.zeros(img_shape)

        # trigger locates in right bottom corner, change corresponding mask values
        self.mask[height-mark_height:][:, width-mark_width:] = 1


        # get average feature map from yt's subset images
        print("...extracting target feature map")

        self.model.cuda()
        _yt_inputs = self.yt_imgs[self.yt_sub_inds].cuda()
        yt_featmaps = self.model.get_fm_before_outlayer(_yt_inputs)   # on cuda
        assert yt_featmaps.shape[0]==len(self.yt_sub_inds), \
               "yt image num mismatch, check 'latent_backdoor' attack-->'generate_trigger' function"
        yt_avg_featmap = yt_featmaps.sum(0) / len(self.yt_sub_inds)


        #------------------------- below is trigger generation process -------------------------#
        print("...generating trigger")

        # self.model.cuda()
        self.mark = self.mark.cuda()
        self.mask = self.mask.cuda()
        yt_avg_featmap = yt_avg_featmap.cuda()
        self.mark.requires_grad = True

        if not self.poison_lr:
            self.poison_lr = 0.01

        optimizer = optim.Adam([self.mark], lr=self.poison_lr)
        criterion = nn.MSELoss()
        self.model.eval()
        for epoch in range(self.poison_iteration):
            optimizer.zero_grad()
            l2_loss = 0.0
            for ind in self.ynt_sub_inds:
                ynt_img = self.ynt_imgs[ind].cuda()
                ynt_img = ynt_img * (1-self.mask) + self.mask * torch.sigmoid(self.mark)
                ynt_img = ynt_img.unsqueeze(0)
                ynt_featmap = self.model.get_fm_before_outlayer(ynt_img)
                l2_loss += criterion(ynt_featmap, yt_avg_featmap)

            l2_loss = torch.div(l2_loss, len(self.ynt_sub_inds))
            print("Epoch {} | MSE Loss {}".format(epoch, l2_loss))
            l2_loss.backward(retain_graph=True)
            optimizer.step()

        self.model.cpu()
        self.mark = self.mark.cpu()
        self.mask = self.mask.cpu()

        # now we generate an universal trigger (self.mark)
        print("trigger generation done!\n")


    def student_fine_tuning(self):
        """
        Assume now a victim get the backdoored model, she remove the org classifier 
        with a new one, then only train this classifier.

        The fine-tuned classifier is last (or last two) layer(s) of model. We only
        fine-tune last layer by default.
        """
        print("replacing outlayer with a clean one")
        train_params = self.model.add_new_last_layer()
        for param in train_params:
            param.requires_grad = True

        yt_num = len(self.yt_labels)
        ynt_num = len(self.ynt_labels)
        all_num = yt_num + ynt_num
        fine_tune_inds = np.random.choice(list(range(all_num)), 
                         int(all_num*self.fine_tune_set_ratio), 
                         replace=False)

        fine_tune_imgs = torch.tensor(self.yt_imgs.tolist()+self.ynt_imgs.tolist())[fine_tune_inds]
        fine_tune_labels = torch.tensor(self.yt_labels.tolist()+self.ynt_labels.tolist())[fine_tune_inds]

        # only update last layer parameters
        optimizer = optim.Adam(train_params, lr=self.poison_lr)
        criterion = nn.CrossEntropyLoss()

        print("...fine-tuning")
        self.model.cuda()
        self.model.train()

        for epoch in range(20):
            optimizer.zero_grad()
            loss, n_sample = 0.0, 0
            for input, label in zip(fine_tune_imgs, fine_tune_labels):
                logits = self.model(input.cuda())
                logits.unsqueeze(0)
                label.unsqueeze(0)
                loss += criterion(logits, label.cuda())
                n_sample += 1
            loss = torch.div(loss, n_sample)
            loss.backward()
            optimizer.step()

        self.model.cpu()
        self.model.eval()


    #-----------------------------------------------------------------------#
    #                    below are some helper functions                    #
    #-----------------------------------------------------------------------#
    def loader_to_dataset(self, loader: torch.utils.data.DataLoader) -> (torch.tensor, torch.LongTensor):
        """
        Extract image tensors with labels from a Dataloader, whatever batch size.
        """
        imgs = []
        labels = []
        for batch_imgs, lbs in loader:
            imgs += batch_imgs.tolist()
            labels += lbs.tolist()
        return torch.tensor(imgs), torch.LongTensor(labels)

