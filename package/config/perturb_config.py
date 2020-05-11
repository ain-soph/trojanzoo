# -*- coding: utf-8 -*-
perturb_params = {
    'pgd': {
        'white': {

            # 97.6%
            'mnist': {'alpha': 0.02, 'epsilon': 0.2, 'iteration': 100},
            # 13 100% 16 99.99%
            'cifar10': {'alpha': 3.0/255, 'epsilon': 8.0/255, 'iteration': 20},
            'gtsrb':{'alpha': 2.0/255, 'epsilon': 4.0/255, 'iteration': 20},
            'isic2018':{'alpha':1.0/255, 'epsilon': 2.0/255, 'iteration': 20},
            'sample_imagenet':{'alpha': 2.0/255, 'epsilon': 4.0/255, 'iteration': 20},
            'dogfish': {'alpha': 0.01, 'epsilon': 0.05, 'iteration': 100},

            # 'mnist': {'alpha': 0.01, 'epsilon': 0.1, 'iteration': 10}, # 59.5%
            # 'cifar10': {'alpha': 0.002, 'epsilon': 0.02, 'iteration': 10} # 13 62.7% 16 53.8%

            # 'mnist': {'alpha': 0.005, 'epsilon': 0.05, 'iteration': 100}, # 97.6% #fpr saving images
        },
        'black': {
            # 73.0%
            'mnist': {'alpha': 0.02, 'epsilon': 0.2, 'iteration': 100},
            # 13 42.7% 16 39.3%
            'cifar10': {'alpha': 0.005, 'epsilon': 0.05, 'iteration': 100},
            'dogfish': {'alpha': 0.01, 'epsilon': 0.05, 'iteration': 100},
        }
    },
    'spatial': {
        'white': {
            'mnist': {'tau': 0.05, 'iteration': 100, 'lr': 0.1},  # 80.16%
            # 13 92.7% 16 93.2%
            'cifar10': {'tau': 0.05, 'iteration': 100, 'lr': 0.1},
            'dogfish': {'tau': 0.05, 'iteration': 100, 'lr': 0.1},

            # 'cifar10': {'tau': 0.05, 'iteration': 20, 'lr': 0.1} # 13 74.5% 16 78.6%
        },
        'black': {
            'mnist': {'tau': 0.05, 'iteration': 100, 'lr': 0.1},  # 74.6%
            # 13 37.3% 16 35.9%
            'cifar10': {'tau': 0.05, 'iteration': 100, 'lr': 0.1},
            'dogfish': {'tau': 0.05, 'iteration': 100, 'lr': 0.1},
        }
    },
    'watermark': {
        'white': {
            'mnist': {'alpha': 0.7, 'pgd_alpha': 0.05, 'pgd_epsilon': 0.8, 'iteration': 1, 'threshold': 5, 'target_value': 10, 'epoch': 100, 'lr': 0.05, 'neuron_num': 2, 'batch_num': 20},
            'cifar10': {'alpha': 0.7, 'pgd_alpha': 0.01, 'pgd_epsilon': 2, 'iteration': 1, 'threshold': 5, 'target_value': 10, 'epoch': 1000, 'lr': 0.05, 'neuron_num': 2, 'batch_num': 20},
            'dogfish': {'alpha': 0.7, 'pgd_alpha': 0.01, 'pgd_epsilon': 2, 'iteration': 1, 'threshold': 10, 'target_value': 100, 'epoch': 1000, 'lr': 0.05, 'neuron_num': 2, 'batch_num': 20},
        },
        'black': {
            'mnist': {'alpha': 0.7, 'pgd_alpha': 0.05, 'pgd_epsilon': 0.8, 'iteration': 1, 'threshold': 10, 'target_value': 100, 'epoch': 100, 'lr': 0.05, 'neuron_num': 2, 'batch_num': 10},
            'cifar10': {'alpha': 0.7, 'pgd_alpha': 0.05, 'pgd_epsilon': 0.8, 'iteration': 1, 'threshold': 10, 'target_value': 100, 'epoch': 100, 'lr': 0.05, 'neuron_num': 2, 'batch_num': 10},
            'dogfish': {'alpha': 0.7, 'pgd_alpha': 0.01, 'pgd_epsilon': 2, 'iteration': 10, 'threshold': 10, 'target_value': 100, 'epoch': 100, 'lr': 0.05, 'neuron_num': 2, 'batch_num': 10},
        }
    },
    'inference': {
        'black': {
            'mnist': {'alpha': 0.05, 'epsilon': 0.3, 'iteration': 10, 'query_num': 50, 'sigma': 0.001, 'fake_percent': 0.4, 'dist': 5.0, 'k': 1, 'b': 0.0085, 'active_percent': 0.1, 'active_multiplier': 1},
            'cifar10': {'alpha': 0.01, 'epsilon': 0.05, 'iteration': 10, 'query_num': 50, 'sigma': 0.001, 'fake_percent': 0.4, 'dist': 5.0, 'k': 1, 'b': 0.004, 'active_percent': 0.1, 'active_multiplier': 1},
            'cifar100': {'alpha': 0.002, 'epsilon': 0.01, 'iteration': 10, 'query_num': 50, 'sigma': 0.001, 'fake_percent': 0.4, 'dist': 5.0, 'k': 1, 'b': 0.004, 'active_percent': 0.1, 'active_multiplier': 1},
            'sample_imagenet': {'alpha': 0.002, 'epsilon': 0.01, 'iteration': 10, 'query_num': 50, 'sigma': 0.001, 'fake_percent': 0.4, 'dist': 5.0, 'k': 1, 'b': 0.004, 'active_percent': 0.1, 'active_multiplier': 1},
            'dogfish': {'alpha': 0.01, 'epsilon': 0.05, 'iteration': 10, 'query_num': 50, 'sigma': 0.001, 'fake_percent': 0.4, 'dist': 5.0, 'k': 1, 'b': 0.004, 'active_percent': 0.1, 'active_multiplier': 1},
            'tiny_imagenet': {'alpha': 0.01, 'epsilon': 0.05, 'iteration': 10, 'query_num': 50, 'sigma': 0.001, 'fake_percent': 0.4, 'dist': 5.0, 'k': 1, 'b': 0.004, 'active_percent': 0.1, 'active_multiplier': 1},
        }
    },
    'unify': {
        'white': {
            'mnist': {},
            'cifar10': {'alpha': 0.0001, 'epsilon': 0.002, 'iteration': 20, 'learning_rate': 0.005, 'retrain_epoch': 10, 'poison_percent': 1},
            'cifar100': {},
            # 'sample_imagenet': {'alpha': 0.0001, 'epsilon': 0.002, 'iteration': 15, 'learning_rate': 0.001, 'retrain_epoch': 10, 'poison_percent': 1},
            'sample_imagenet': {'alpha': 0.00005, 'epsilon': 0.001, 'iteration': 20, 'learning_rate': 0.001, 'retrain_epoch': 10, 'poison_percent': 1},
            'isic2018': {'alpha': 0.00005, 'epsilon': 0.001, 'iteration': 20, 'learning_rate': 0.001, 'retrain_epoch': 10, 'poison_percent': 1},
          #  'gtsrb': {'alpha': 0.001, 'epsilon': 0.02, 'iteration': 15, 'learning_rate': 0.01, 'retrain_epoch': 10, 'poison_percent': 1},   # default
            # 'gtsrb': {'alpha': 0.01, 'epsilon': 0.2, 'iteration': 15, 'learning_rate': 0.001, 'retrain_epoch': 10, 'poison_percent': 1},
            'gtsrb': {'alpha': 0.001, 'epsilon': 0.02, 'iteration': 20, 'learning_rate': 0.005, 'retrain_epoch': 10, 'poison_percent': 1},
            'tiny_imagenet': {},
        }
    },
}


def get_perturb_params(input_method, dataset, mode='white', **kwargs):
    if mode not in ['white','black']:
        mode='white'
    if dataset not in perturb_params[input_method][mode].keys():
        dataset='cifar10'
    return perturb_params[input_method][mode][dataset]
