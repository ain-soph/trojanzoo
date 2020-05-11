# -*- coding: utf-8 -*-
from .dataset import Dataset
from package.utils.utils import to_tensor

class GraphSet(Dataset):
    """docstring for dataset"""

    def __init__(self, name='imageset', **kwargs):
        super(GraphSet, self).__init__(name=name, data_type='graph', **kwargs)

    @staticmethod
    def get_data(data):
        _dict={}
        _dict['x']=to_tensor(data.x)
        _dict['edge_index']=to_tensor(data.edge_index, dtype='int')
        
        return _dict, to_tensor(data.y, dtype='long')
