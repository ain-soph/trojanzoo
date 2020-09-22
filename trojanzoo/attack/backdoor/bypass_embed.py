import torch
import torch.nn as nn
import torch.optim as optim

from collections import OrderedDict
from torch.utils.data import Dataset, DataLoader
from .badnet import BadNet

class Bypass_Embed(BadNet):
    name: str = 'bypass_embed'

    def __init__(self, lambd: int = 10, discrim_lr: float = 0.001,
                 **kwargs):
        super().__init__(**kwargs)

        self.param_list['bypass_embed'] = ['lambd', 'discrim_lr']

        self.lambd = lambd
        self.discrim_lr = discrim_lr

    def attack(self, **kwargs):
        print('Sample Data')
        trainloader = self.sample_data()  # with poisoned images
        print('Joint Training')
        self.joint_train(trainloader)

    def sample_data(self):
        other_classes = list(range(self.dataset.num_classes))
        other_classes.pop(self.target_class)
        other_x, other_y = [], []
        for _class in other_classes:
            loader = self.dataset.get_dataloader(mode='train', batch_size=self.class_sample_num, classes=[_class],
                                                 shuffle=True, num_workers=0, pin_memory=False)
            _input, _label = next(iter(loader))
            other_x.append(_input)
            other_y.append(_label)
        other_x = torch.cat(other_x)
        other_y = torch.cat(other_y)

        poison_x = self.mark.add_mark(other_x)
        poison_y = torch.zeros_like(other_y)
        poison_y.fill_(self.target_class)

        clean_x, clean_y = self.dataset.get_dataset(mode='train')
        all_x = torch.cat((clean_x, poison_x))
        all_y = torch.cat((clean_y, poison_y))
        all_discrim_y = torch.cat((torch.zeros_like(clean_y), 
                                   torch.ones_like(poison_y)))
        # used for training
        poison_trainset = TwoLabelsDataset(all_x, all_y, all_discrim_y)
        poison_trainloader = DataLoader(poison_trainset, batch_size=self.batch_size, shuffle=True, pin_memory=False)

        return poison_trainloader

    def joint_train(self, poison_trainloader):
        # get in_dim
        batch_x = poison_trainloader[0] # sample one batch
        batch_x = self.model._model.get_fm(batch_x)
        batch_x = self.model._model.pool(batch_x)
        batch_x = self.model._model.flatten(batch_x)
        in_dim = batch_x.shape[-1]

        D = Discriminator(in_dim)

        device = torch.device('cpu')
        if next(self.model.parameters()).is_cuda:
            device = torch.device('cuda')

        optimizer = optim.Adam([{'params': self.model.parameters()},
                                {'params': D.parameters(), 'lr': self.discrim_lr}
                               ], lr=self.lr)
        if self.lr_scheduler:
            _lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=self.step_size, gamma=0.1)
        loss_fn_d = nn.CrossEntrypyLoss()

        D.to(device)
        for _epoch in range(self.epoch):
            for data in poison_trainloader:
                optimizer.zero_grad()
                _input, _label_f, _label_d = self.dataset.get_data(data, mode='train')
                out_f = self.model(_input.to(device))
                loss_f = self.loss(out_f, _label_f.to(device))

                # discriminator loss
                batch_x = self.model._model.get_fm(_input)
                batch_x = self.model._model.pool(batch_x)
                batch_x = self.model._model.flatten(batch_x)
                out_d = D(batch_x)

                loss_d = loss_fn_d(out_d, _label_d.to(device))

                loss = loss_f + self.lambd * loss_d
                loss.backward()
                _lr_scheduler.step()

        # no validate yet
        
#----------------------------------------------------------------------------------#

class TwoLabelsDataset(Dataset):
    def __init__(self, data: torch.FloatTensor, labels_1: torch.LongTensor, labels_2: torch.LongTensor):
        '''
        A customized Dataset class with two gruop of labels,
        used for bypass detection backdoor attack ('bypass_embed')
        '''
        self.data = data
        self.labels_1 = labels_1
        self.labels_2 = labels_2

    def __getitem__(self, index):
        x = self.data[index]
        y1 = self.labels_1[index]
        y2 = self.labels_2[index]
        return x, y1, y2

    def __len__(self):
        return len(self.data)


class Discriminator(nn.Module):
    def __init__(self, in_dim: int):
            super(nn.Module).__init__()
            self.D = nn.Sequential(OrderedDict([
                    ('fc1', nn.Linear(in_dim, 256)),
                    ('relu1', nn.ReLU()),
                    ('fc2', nn.Linear(256, 128)),
                    ('relu2', nn.ReLU()),
                    ('fc3', nn.Linear(128, 2)),
                    ('softmax', nn.Softmax())
                    ]))

    def forward(self, x):
        return self.D(x)