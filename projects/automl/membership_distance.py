#!/usr/bin/env python3

#python scripts/membership.py --model densenet121_comp --dataset cifar100 --pretrain

from trojanzoo.utils.tensor import to_numpy
import trojanvision.environ
import trojanvision.datasets
import trojanvision.models
from trojanvision.utils import summary
import argparse
import warnings
import os
warnings.filterwarnings("ignore")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    trojanvision.environ.add_argument(parser)
    trojanvision.datasets.add_argument(parser)
    trojanvision.models.add_argument(parser)
    args = parser.parse_args()
    env = trojanvision.environ.create(**args.__dict__)
    dataset = trojanvision.datasets.create(**args.__dict__)
    model = trojanvision.models.create(dataset=dataset, **args.__dict__)
    if env['verbose']:
        summary(env=env, dataset=dataset, model=model)
    import torch
    import numpy as np
    from sklearn import metrics
    from trojanzoo.utils.data import dataset_to_list
    from art.estimators.classification import PyTorchClassifier  # type: ignore
    classifier = PyTorchClassifier(
        model=model._model,
        loss=model.criterion,
        input_shape=dataset.data_shape,
        nb_classes=model.num_classes,
    )
    model._validate()
    
    from art.attacks.inference.membership_inference import LabelOnlyDecisionBoundary as Attack  # type: ignore

    attack = Attack(classifier)
    x_train, y_train = dataset_to_list(dataset.get_dataset('train'))
    x_train, y_train = to_numpy(torch.stack(x_train)), to_numpy(y_train)
    x_valid, y_valid = dataset_to_list(dataset.get_dataset('valid'))
    x_valid, y_valid = to_numpy(torch.stack(x_valid)), to_numpy(y_valid)

    sample_size = 64

    tau_path = os.path.normpath(os.path.join(model.folder_path, model.name)) + '_tau' 

    t_idx = np.arange(len(x_train))
    v_idx = np.arange(len(x_valid))

    np.random.shuffle(t_idx)
    np.random.shuffle(v_idx)
    
    if os.path.exists(tau_path):
        attack.distance_threshold_tau = np.load(tau_path + '.npy', allow_pickle=True)['tau']
    else:
        train_idx = t_idx[:sample_size]
        valid_idx = v_idx[:sample_size]

        attack.calibrate_distance_threshold(x_train=x_train[train_idx], y_train=y_train[train_idx], x_test=x_valid[valid_idx], y_test=y_valid[valid_idx], max_iter=10, max_eval=500, norm = np.inf, verbose=False)
        np.save(tau_path, {'tau': attack.distance_threshold_tau})

    train_idx = t_idx[sample_size:2*sample_size]
    valid_idx = v_idx[sample_size:2*sample_size]

    x_train, y_train = x_train[train_idx], y_train[train_idx]
    x_valid, y_valid = x_valid[valid_idx], y_valid[valid_idx]

    result = np.concatenate((attack.infer(x=x_train, y=y_train, max_iter=10, max_eval=500, norm = np.inf, verbose=False), attack.infer(x=x_valid, y=y_valid, max_iter=10, max_eval=500, norm = np.inf, verbose=False)))
    y_truth = np.concatenate(([1] * len(x_train), [0] * len(x_valid)))
    print('F1 score: ', metrics.f1_score(result, y_truth))
    print('Accuracy score: ', metrics.accuracy_score(result, y_truth))
    print('Recall score: ', metrics.recall_score(result, y_truth))
    print('Precision score: ', metrics.precision_score(result, y_truth))
