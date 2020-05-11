# -*- coding: utf-8 -*-
from .gnn_node import GNN_Node


class GNN_Baseline(GNN_Node):

    def __init__(self, name='gcn_baseline', data_dir='./data/', dataset='TUDataset', num_classes=2, num_features=7, conv_depth=5, **kwargs):
        super(GNN_Baseline, self).__init__(name=name, data_dir=data_dir, dataset=dataset, num_classes=num_classes,
                                           num_features=num_features, conv_depth=conv_depth, conv_dim=num_features, fc_depth=0, fc_dim=0,  **kwargs)

    def forward(self, x, edge_index, mask):
        x = self.get_fm(x, edge_index)
        return x[mask]
