from trojanzoo.parser import Parser_Dataset, Parser_Model, Parser_Train, Parser_Seq, Parser_Mark, Parser_Attack, Parser_Defense

from trojanzoo.dataset import ImageSet
from trojanzoo.model import ImageModel
from trojanzoo.mark import Watermark
from trojanzoo.attack.backdoor import BadNet
from trojanzoo.utils.tensor import to_tensor

import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler

import warnings
warnings.filterwarnings("ignore")

# python transfer_attack.py --parameters classifier
if __name__ == '__main__':

    parser = Parser_Seq(Parser_Dataset(), Parser_Model(), Parser_Train(),
                        Parser_Mark(), Parser_Attack())
    parser.parse_args()
    parser.get_module()

    dataset: ImageSet = parser.module_list['dataset']
    model: ImageModel = parser.module_list['model']
    mark: Watermark = parser.module_list['mark']
    attack: BadNet = parser.module_list['attack']
    attack.load()

    attack.validate_func()

    train_feat = []
    train_label = []
    valid_feat = []
    valid_label = []
    poison_feat = []
    poison_label = []
    for data in dataset.loader['train']:
        _input, _label = model.get_data(data)
        _feat = model.get_final_fm(_input)

        train_feat.append(_feat.detach().cpu())
        train_label.append(_label.detach().cpu())
    for data in dataset.loader['valid']:
        _input, _label = model.get_data(data)
        _feat = model.get_final_fm(_input)
        valid_feat.append(_feat.detach().cpu())
        valid_label.append(_label.detach().cpu())

        poison_input = attack.add_mark(_input)
        _poison_label = attack.target_class * torch.ones_like(_label)
        _poison_feat = model.get_final_fm(poison_input)
        poison_feat.append(_poison_feat.detach().cpu())
        poison_label.append(_poison_label.detach().cpu())
    train_feat = torch.cat(train_feat).detach().numpy()
    train_label = torch.cat(train_label).detach().numpy()
    valid_feat = torch.cat(valid_feat).detach().numpy()
    valid_label = torch.cat(valid_label).detach().numpy()
    poison_feat = torch.cat(poison_feat).detach().numpy()
    poison_label = torch.cat(poison_label).detach().numpy()

    model_dict = {
        'bayes': GaussianNB(),
        'svm': SVC(),
        'random_forest': RandomForestClassifier()
    }
    for name, sk_model in model_dict.items():
        sk_model.fit(train_feat, train_label)
        clean_acc = sk_model.score(valid_feat, valid_label) * 100
        poison_acc = sk_model.score(poison_feat, poison_label) * 100
        print(f'{name:15s} clean acc: {clean_acc:7.3f} poison acc: {poison_acc:7.3f}')
