# -*- coding: utf-8 -*-
from package.imports.universal import *
from package.utils.utils import to_numpy, to_tensor
from package.utils.output import prints
# from package.utils.main_utils import get_model


class Dataset(object):
    """docstring for dataset"""

    def __init__(self, name='abstact_dataset', data_type='abstract', data_dir='./data/', result_dir='./result/', memory_dir: str = None, folder_path: str = None,
                 batch_size=128, num_classes: int = None, test_set=False, loss_weights=False,
                 torch_seed=1228, numpy_seed=100, train_num=1024, num_workers=4, default_model='default', indent=0, output=True, **kwargs):
        self.name = name
        self.data_type = data_type
        self.data_dir = data_dir
        self.result_dir = result_dir
        self.indent = indent
        self.output = output
        if memory_dir is not None:
            if not os.path.exists(memory_dir+self.data_type+'/'+self.name+'/data/'):
                memory_dir = None
                # raise ValueError('\"memory_dir\" not exist: '+memory_dir)
        self.folder_path = folder_path
        if self.folder_path is None:
            if memory_dir is not None:
                self.folder_path = memory_dir+self.data_type+'/'+self.name+'/data/'
            else:
                self.folder_path = self.data_dir+self.data_type+'/'+self.name+'/data/'
        if not os.path.exists(self.folder_path):
            os.makedirs(self.folder_path)

        self.torch_seed = torch_seed
        self.numpy_seed = numpy_seed
        self.train_num = train_num
        self.num_workers = num_workers

        self.batch_size = batch_size
        self.num_classes = num_classes
        self.test_set = test_set
        if isinstance(loss_weights, bool):
            if loss_weights:
                self.loss_weights = self.get_loss_weights()
            else:
                self.loss_weights = None
        else:
            self.loss_weights = loss_weights

        self.num_gpus = torch.cuda.device_count()

        self.default_model = default_model

        self.loader = {}
        # batch_size = self.num_gpus*self.batch_size

        self.loader['train'] = self.get_dataloader(
            mode='train', batch_size=batch_size, full=True)
        self.loader['valid'] = self.get_dataloader(
            mode='valid', batch_size=batch_size, full=True)
        self.loader['train2'] = self.get_dataloader(
            mode='train', batch_size=batch_size, full=False)
        self.loader['valid2'] = self.get_dataloader(
            mode='valid', batch_size=batch_size, full=False)
        self.loader['test'] = self.get_dataloader(mode='test', batch_size=1)

        self.output_par(name='Dataset')

    def initialize(self):
        pass

    def output_par(self, name: str = None, _filter=[], indent: int = None, output: bool = None):
        if output is None:
            output = self.output
        if not output:
            return
        if name is not None:
            if name != self.__class__.__name__:
                return
        if indent is None:
            indent = self.indent
        prints(self.name.rjust(10)+' Parameters: ', indent=indent)
        d = self.__dict__
        _dict = {}
        for key in d.keys():
            if '__' not in key and 'function' not in type(d[key]).__name__ and 'method' not in type(d[key]).__name__ and 'loader' not in key and 'seed' not in key and key not in ['data_type', 'num_gpus', 'num_workers', 'url', 'org_folder_name'] and key not in _filter:
                _dict[key] = d[key]
        prints(_dict, indent=indent)
        print()

    def get_transform(self, mode):
        pass

    @staticmethod
    def get_data(data):
        return data

    def get_full_dataset(self, mode, transform: object = None):
        return []

    def get_dataset(self, mode: str, full=True, **kwargs):
        if full:
            return self.get_full_dataset(mode)
        else:
            if mode == 'train':
                full_dataset = self.get_full_dataset(mode)
                indices = list(range(len(full_dataset)))
                np.random.seed(self.numpy_seed)
                np.random.shuffle(indices)
                return torch.utils.data.Subset(full_dataset, indices[:self.train_num])
            else:
                return self.get_split_validset(mode, **kwargs)

    def get_dataloader(self, mode: str, full=False, batch_size: int = None, shuffle: bool = None, num_workers: int = None, **kwargs) -> torch.utils.data.dataloader:
        return []

    def get_split_validset(self, mode: str, valid_percent=0.6) -> torch.utils.data.dataloader:
        if self.test_set:
            return self.get_full_dataset(mode)
        full_dataset = self.get_full_dataset('valid')
        split = int(np.floor(valid_percent * len(full_dataset)))
        indices = list(range(len(full_dataset)))
        np.random.seed(self.numpy_seed)
        np.random.shuffle(indices)
        if mode == 'test':
            return torch.utils.data.Subset(full_dataset, indices[split:])
        elif mode == 'valid':
            return torch.utils.data.Subset(full_dataset, indices[:split])
        else:
            raise ValueError(
                'argument \"mode\" value must be \"valid\" or \"test\"!')

    def get_loss_weights(self, file_path: str = None) -> torch.FloatTensor:
        if file_path is None:
            file_path = self.folder_path+'loss_weights.npy'
        if os.path.exists(file_path):
            loss_weights = to_tensor(np.load(file_path), dtype='float')
            return loss_weights
        else:
            print('Calculating Loss Weights')
            train_loader = self.get_dataloader('train', full=True)
            loss_weights = np.zeros(self.num_classes)
            for i, (X, Y) in enumerate(train_loader):
                Y = to_numpy(Y).tolist()
                for _class in range(self.num_classes):
                    loss_weights[_class] += Y.count(_class)
            loss_weights = loss_weights.sum() / loss_weights
            np.save(file_path, loss_weights)
            print('Loss Weights Saved at ', file_path)
            return to_tensor(loss_weights, dtype='float')

    # def get_model(self, model_name=None, *args, **kwargs):
    #     if model_name is None:
    #         model_name = self.default_model
    #     get_model(model_name, *args, **kwargs)
