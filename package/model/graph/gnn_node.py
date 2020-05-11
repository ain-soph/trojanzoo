# -*- coding: utf-8 -*-
from .gnn import GNN


class GNN_Node(GNN):

    def __init__(self, name='gnn_node', data_dir='./data/', dataset='TUDataset', num_classes=2, num_features=7, conv_depth=2, conv_dim=32, **kwargs):
        super(GNN_Node, self).__init__(name=name, data_dir=data_dir, dataset=dataset, num_classes=num_classes,
                                       num_features=num_features, conv_depth=conv_depth, conv_dim=conv_dim, fc_depth=0, fc_dim=0,  **kwargs)

    def forward(self, x, edge_index, mask):
        x = self.get_fm(x, edge_index)
        return x[mask]
