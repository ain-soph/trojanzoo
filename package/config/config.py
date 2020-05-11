# -*- coding: utf-8 -*-
data_configs = {

    'cifar10': {
        'n_dim': 32,
        'n_channel': 3,
        'num_classes': 10,
        'model_name': 'resnetnew18'
    },
    'cifar100': {
        'n_dim': 32,
        'n_channel': 3,
        'num_classes': 100,
        'model_name': 'resnetnew18'
    },

    'mnist': {
        'n_dim': 28,
        'n_channel': 1,
        'num_classes': 10,
        'model_name': 'lenet'
    },
    'imagenet': {
        'n_dim': 224,
        'n_channel': 3,
        'num_classes': 1000,
        'model_name': 'resnet'
    },
    'sample_imagenet': {
        'n_dim': 224,
        'n_channel': 3,
        'num_classes': 20,
        'model_name': 'resnet18'
    },
    'dogfish': {
        'n_dim': 224,
        'n_channel': 3,
        'num_classes': 2,
        'model_name': 'resnet'
    },
    'tiny_imagenet': {
        'n_dim': 224,
        'n_channel': 3,
        'num_classes': 200,
        'model_name': 'resnetnew18'
    },

    'isic2018': {
        'n_dim': 224,
        'n_channel': 3,
        'num_classes': 7,
        'model_name': 'resnet101',
    },
    'isic2019': {
        'n_dim': 224,
        'n_channel': 3,
        'num_classes': 9,
        'model_name': 'resnetnew18',
    },
    'gtsrb': {
        'n_dim': 32,
        'n_channel': 3,
        'num_classes': 43,
        'model_name': 'resnetnew18',
        'weight': True,
    },


    'mutag': {
        'num_classes': 2,
        'num_features': 7,
        'model_name': 'gin'
    },
    'cora': {
        'num_classes': 7,
        'num_features': 1433,
        'model_name': 'gin_node'
    },
    'citeseer': {
        'num_classes': 6,
        'num_features': 3703,
        'model_name': 'gin_node'
    },
    'pubmed': {
        'num_classes': 3,
        'num_features': 500,
        'model_name': 'gin_node'
    },
}

name2class_map = {
    'lenet': 'LeNet',
    'lenet_simple': 'LeNet_Simple',
    'vgg': 'VGG',
    'vggcomp': 'VGGcomp',
    'resnet': 'ResNet',
    'resnetnew': 'ResNetNew',
    'resnet_finetune': 'ResNet_FineTune',
    'gnn': 'GNN',
    'gnn_node': 'GNN_Node',
    'gnn_baseline': 'GNN_Baseline',
    'gnn_reverse': 'GNN_Reverse',

    'pgd': 'PGD',
    'spatial': 'Spatial',
    'watermark': 'Watermark',
    'inference': 'Inference',
    'unify': 'Unify',

    'magnet': 'MagNet',
    'lid': 'LID',

}


def get_data_params(dataset):
    return data_configs[dataset]


def get_model_name(name):
    if len(name) > 2:
        if name[0] == 'g' and name[2] == 'n':
            name = name[:1]+'n'+name[2:]
    return name


def name2class(name):
    return name2class_map[name]
