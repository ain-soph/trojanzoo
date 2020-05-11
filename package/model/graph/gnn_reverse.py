# -*- coding: utf-8 -*-
from model.gnn_node import *


class GNN_Reverse(GNN_Node):

    def __init__(self, name='gcn_reverse', data_dir='./data/', dataset='TUDataset', num_classes=7, num_features=1433, conv_depth=5, **kwargs):
        super(GNN_Reverse, self).__init__(name=name, data_dir=data_dir, dataset=dataset, num_classes=num_features,
                                          num_features=num_features, conv_depth=conv_depth, conv_dim=num_features,  **kwargs)
        self.num_classes = num_classes

    def forward(self, x, edge_index, mask):
        x = self.get_fm(x, edge_index)
        return x[mask][:, 0:self.num_classes]

    def define_features(self):
        conv_seq = []
        half = self.num_features-self.num_features//2
        for i in range(self.conv_depth):
            conv_seq.append(('conv'+str(i+1)+'f', self.conv_layer(half, half)))
            conv_seq.append(('conv'+str(i+1)+'g', self.conv_layer(half, half)))
            conv_seq.append(('relu'+str(i+1), nn.ReLU()))
            conv_seq.append(('dropout'+str(i+1), nn.Dropout()))
            # conv_seq.append(('bn'+str(i+1), nn.BatchNorm1d(self.conv_dim)))
        conv_seq = OrderedDict(conv_seq)
        self.features = GNNfeatures_Reverse(conv_seq)
        return GNNfeatures(conv_seq)


class GNNfeatures_Reverse(GNNfeatures):
    def __init__(self, conv_seq):
        super(GNNfeatures_Reverse, self).__init__(conv_seq)

    def forward(self, x, edge_index):
        padding = False
        for layer_name, layer in self.named_children():
            if 'conv' and 'f' in layer_name:
                if x.size(1) % 2 == 1:
                    padding = True
                    x = to_tensor(
                        torch.cat((x, to_tensor(torch.zeros(x.size(0), 1))), dim=1))
                x1 = x[:, 0:x.size(1)//2]
                x2 = x[:, x.size(1)//2:]
                x1 = x1 + layer(x2, edge_index)
            elif 'conv' and 'g' in layer_name:
                x2 = x2 + layer(x1, edge_index)
                x = to_tensor(torch.cat((x1, x2), dim=1))
                if padding:
                    x = x[:, :-1]
                    padding = False
            else:
                x = layer(x)
        return x
