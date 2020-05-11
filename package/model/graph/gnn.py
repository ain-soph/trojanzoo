# -*- coding: utf-8 -*-
from ..model import Model
from ...imports.universal import *
from collections import OrderedDict 
from torch_geometric.nn import global_mean_pool, GCNConv, GINConv


class GNN(Model):

    def __init__(self, name='gnn', data_dir='./data/', dataset='TUDataset', num_classes=2, num_features=7, conv_depth=5, conv_dim=32, fc_depth=2, fc_dim=32, **kwargs):
        super(GNN, self).__init__(name=name, data_dir=data_dir, dataset=dataset, num_classes=num_classes,
                                  conv_depth=conv_depth, conv_dim=conv_dim, fc_depth=fc_depth, fc_dim=fc_dim,  **kwargs)
        self.num_features = num_features
        self.features = self.define_features()

    def forward(self, x, edge_index, batch):
        x = self.get_fm(x, edge_index)
        x = global_mean_pool(x, batch)
        x = self.get_logits_from_fm(x)
        return x

    def get_fm(self, x, edge_index):
        return self.features(x, edge_index)

    def conv_layer(self, l, r):
        if 'gcn' in self.name:
            return GCNConv(l, r)
        elif 'gin' in self.name:
            kernel = nn.Sequential(OrderedDict(
                [('fc1', nn.Linear(l, l)), ('relu1', nn.ReLU()), ('fc2', nn.Linear(l, r))]))
            return GINConv(kernel, train_eps=False)

    # def define_optimizer(self, train_opt='partial', lr=None, optim_type=None, lr_scheduler=False, **kwargs):
    #     return super(GIN, self).define_optimizer(train_opt=train_opt, lr=0.01, optim_type='Adam', lr_scheduler=lr_scheduler, weight_decay=5e-3)

    def define_features(self):
        conv_seq = []
        for i in range(self.conv_depth):
            l = self.num_features if i == 0 else self.conv_dim
            r = self.num_classes if i == self.conv_depth - \
                1 and self.fc_depth == 0 else self.conv_dim
            conv_seq.append(('conv'+str(i+1), self.conv_layer(l, r)))
            conv_seq.append(('relu'+str(i+1), nn.ReLU()))
            conv_seq.append(('bn'+str(i+1), nn.BatchNorm1d(r)))
        conv_seq = OrderedDict(conv_seq)
        return GNNfeatures(conv_seq)

    def define_optimizer(self, train_opt='full', lr=0.01, weight_decay=5e-4, **kwargs):
        optimizer = super(GNN, self).define_optimizer(
            train_opt=train_opt, optim_type='Adam', lr=lr, weight_decay=weight_decay, **kwargs)
        return optimizer


class GNNfeatures(nn.Module):
    def __init__(self, conv_seq):
        super(GNNfeatures, self).__init__()
        for layer_name in conv_seq:
            setattr(self, layer_name, conv_seq[layer_name])

    def forward(self, x, edge_index):
        for layer_name, layer in self.named_children():
            if 'conv' in layer_name:
                x = layer(x, edge_index)
            else:
                x = layer(x)
        return x
