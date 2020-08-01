from trojanzoo.parser import Parser_Dataset, Parser_Model, Parser_Train, Parser_Seq, Parser_Mark, Parser_Attack, Parser_Defense

from trojanzoo.dataset import ImageSet
from trojanzoo.model import ImageModel
from trojanzoo.utils.mark import Watermark
from trojanzoo.attack.backdoor import BadNet
from trojanzoo.defense import Defense_Backdoor

import torch
import torch.nn as nn
import torch.optim as optim

import warnings
warnings.filterwarnings("ignore")


class TransferAttack:
    def __init__(self, backdoor_model_path: str, trigger_path: str, data_name: str, 
                epoch: int =100, lr: float=0.01):
        self.model = self.load_model(backdoor_model_path)
        self.mark, self.mask, self.clean_data = self.load_data(trigger_path, data_name)

        self.epoch = epoch
        self.lr = lr

    def load_model(self, path: str) -> ImageModel:
        # load backdoored model
        pass

    def load_data(self, path: str, data_name: str) -> (torch.Tensor, torch.Tensor, torch.Tensor):
        # load mark, mask and clean data
        pass

    def retrain_with_org_classifier(self):
        self.model._model.classifier.apply(self.weight_init)

        optimizer = optim.Adam(self.model._model.classifier.parameters(), lr=self.lr)
        self.model.model._train(self.epoch, validate_func=self.validate_func, optimizer=optimizer)


    def retrain_with_new_classifier(self, fc_depth: int):
        conv_dim = self.model._model.conv_dim
        fc_dim = self.model._model.fc_dim
        num_classes = self.model._model.num_classes
        self.model._model.classifier = self.model._model.define_classifier(num_classes, conv_dim, fc_depth, fc_dim)
        
        optimizer = optim.Adam(self.model._model.classifier.parameters(), lr=self.lr)
        self.model.model._train(self.epoch, validate_func=self.validate_func, optimizer=optimizer)


    def fine_tuning(self, mode='classifier'):
        if mode == 'full':
            optimizer = optim.Adam(self.model._model.parameters(), lr=self.lr)
        elif mode == 'classifier':
            optimizer = optim.Adam(self.model._model.classifier.parameters(), lr=self.lr)
        self.model.model._train(self.epoch, validate_func=self.validate_func, optimizer=optimizer)
    
    
    def validate_func(self, get_data=None, loss_fn=None, **kwargs) -> (float, float, float):
        clean_loss, clean_acc, _ = self.model.model._validate(print_prefix='Validate Clean',
                                                        get_data=None, **kwargs)
        target_loss, target_acc, _ = self.model.model._validate(print_prefix='Validate Trigger Tgt',
                                                          get_data=self.get_data, keep_org=False, **kwargs)
        _, orginal_acc, _ = self.model.model._validate(print_prefix='Validate Trigger Org',
                                                 get_data=self.get_data, keep_org=False, poison_label=False, **kwargs)
        if self.clean_acc - clean_acc > 3 and self.clean_acc > 40:
            target_acc = 0.0
        return clean_loss + target_loss, target_acc, clean_acc


if __name__ == '__main__':
    parser = Parser_Seq(Parser_Dataset(), Parser_Model(), Parser_Train(),
                        Parser_Mark(), Parser_Attack(), Parser_Defense())
    parser.parse_args()
    parser.get_module()

    dataset: ImageSet = parser.module_list['dataset']
    model: ImageModel = parser.module_list['model']
    optimizer, lr_scheduler, train_args = parser.module_list['train']
    mark: Watermark = parser.module_list['mark']
    attack: BadNet = parser.module_list['attack']

    try: 
        fc_depth: int = parser.module_list['fc_depth']
    except:
        fc_depth = None
    transfer_attack = parser.module_list['transfer_attack'] # todo

    transfer_attack.retrain_with_org_classifier()
    transfer_attack.retrain_with_new_classifier(fc_depth)
    transfer_attack.fine_tuning()