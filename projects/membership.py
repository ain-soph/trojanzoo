#!/usr/bin/env python3

from art.attacks.inference.membership_inference import MembershipInferenceBlackBox as Attack  # type: ignore
from trojanzoo.utils.tensor import to_numpy
import trojanvision.environ
import trojanvision.datasets
import trojanvision.models
from trojanvision.utils import summary
import argparse

import warnings
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

    # from art.attacks.inference.membership_inference import MembershipInferenceBlackBoxRuleBased as Attack
    # attack = Attack(classifier)
    # x_train, y_train = dataset_to_list(dataset.get_dataset('train'))
    # x_train, y_train = to_numpy(torch.stack(x_train)), to_numpy(y_train)
    # x_valid, y_valid = dataset_to_list(dataset.get_dataset('valid'))
    # x_valid, y_valid = to_numpy(torch.stack(x_valid)), to_numpy(y_valid)
    # result = np.concatenate((attack.infer(x_train, y_train), attack.infer(x_valid, y_valid)))
    # y_truth = np.concatenate(([1] * len(x_train), [0] * len(x_valid)))
    # print('result:')
    # print('F1 score: ', metrics.f1_score(result, y_truth))
    # print('Accuracy score: ', metrics.accuracy_score(result, y_truth))
    # print('Recall score: ', metrics.recall_score(result, y_truth))
    # print('Precision score: ', metrics.precision_score(result, y_truth))

    # from art.attacks.inference.membership_inference import LabelOnlyDecisionBoundary as Attack
    # attack = Attack(classifier)
    # x_train, y_train = dataset_to_list(dataset.get_dataset('train'))
    # x_train, y_train = to_numpy(torch.stack(x_train)), to_numpy(y_train)
    # x_valid, y_valid = dataset_to_list(dataset.get_dataset('valid'))
    # x_valid, y_valid = to_numpy(torch.stack(x_valid)), to_numpy(y_valid)

    # attack.calibrate_distance_threshold(x_train[100:300], y_train[100:300], x_valid[100:300], y_valid[100:300])
    # result = np.concatenate((attack.infer(x_train[:100], y_train[:100]), attack.infer(x_valid[:100], y_valid[:100])))
    # y_truth = np.concatenate(([1] * len(x_train[:100]), [0] * len(x_valid[:100])))
    # print('result:')
    # print('F1 score: ', metrics.f1_score(result, y_truth))
    # print('Accuracy score: ', metrics.accuracy_score(result, y_truth))
    # print('Recall score: ', metrics.recall_score(result, y_truth))
    # print('Precision score: ', metrics.precision_score(result, y_truth))
    attack = Attack(classifier)
    x_train, y_train = dataset_to_list(dataset.get_dataset('train'))
    x_train, y_train = to_numpy(torch.stack(x_train)), to_numpy(y_train)
    x_valid, y_valid = dataset_to_list(dataset.get_dataset('valid'))
    x_valid, y_valid = to_numpy(torch.stack(x_valid)), to_numpy(y_valid)

    x_train, y_train = x_train[:1000], y_train[:1000]
    x_valid, y_valid = x_valid[:1000], y_valid[:1000]

    attack.fit(x_train[100:], y_train[100:], x_valid[100:], y_valid[100:])
    result = np.concatenate((attack.infer(x_train[:100], y_train[:100]), attack.infer(x_valid[:100], y_valid[:100])))
    y_truth = np.concatenate(([1] * len(x_train[:100]), [0] * len(x_valid[:100])))
    print('result:')
    print('F1 score: ', metrics.f1_score(result, y_truth))
    print('Accuracy score: ', metrics.accuracy_score(result, y_truth))
    print('Recall score: ', metrics.recall_score(result, y_truth))
    print('Precision score: ', metrics.precision_score(result, y_truth))
