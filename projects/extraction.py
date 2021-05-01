#!/usr/bin/env python3

import trojanvision
from trojanvision.utils import summary, to_numpy
from trojanzoo.utils.data import dataset_to_list

import torch
import torch.nn as nn
import argparse

import warnings
warnings.filterwarnings("ignore")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    trojanvision.environ.add_argument(parser)
    trojanvision.datasets.add_argument(parser)
    trojanvision.models.add_argument(parser)
    trojanvision.trainer.add_argument(parser)
    args = parser.parse_args()

    env = trojanvision.environ.create(**args.__dict__)
    dataset = trojanvision.datasets.create(**args.__dict__)
    model = trojanvision.models.create(dataset=dataset, **args.__dict__)

    if env['verbose']:
        summary(env=env, dataset=dataset, model=model)
    model._validate()
    print('\n\n')

    from art.estimators.classification import PyTorchClassifier  # type: ignore
    classifier = PyTorchClassifier(
        model=model._model,
        loss=model.criterion,
        input_shape=dataset.data_shape,
        nb_classes=model.num_classes,
    )
    x_train, y_train = dataset_to_list(dataset.get_dataset('train'))
    x_train, y_train = to_numpy(torch.stack(x_train)), to_numpy(y_train)

    # valid_train, valid_valid = dataset.split_set(dataset.get_dataset('valid'), length=5000)
    # x_train, y_train = dataset_to_list(valid_train)
    # x_train, y_train = to_numpy(torch.stack(x_train)), to_numpy(y_train)
    # valid_loader = dataset.get_dataloader('valid', dataset=valid_valid)

    # thieved_model._validate(print_prefix='Before Stealing', loader=valid_loader)
    # thieved_model._validate(print_prefix='After Stealing', loader=valid_loader)

    import art.attacks.extraction  # type:ignore
    for name in ['CopycatCNN', 'KnockoffNets']:
        for use_probability in [True, False]:
            print(name + (' Probabilistic' if use_probability else ' Argmax'))
            AttackClass = getattr(art.attacks.extraction, name)
            _dict = args.__dict__ | {'model_name': 'resnet18_comp', 'pretrain': False, 'official': False}
            thieved_model = trojanvision.models.create(dataset=dataset, **_dict)
            trainer = trojanvision.trainer.create(dataset=dataset, model=thieved_model, **args.__dict__)

            thieved_model.train()
            params: list[nn.Parameter] = []
            for param_group in trainer.optimizer.param_groups:
                params.extend(param_group['params'])
            thieved_model.activate_params(params)

            thieved_classifier = PyTorchClassifier(
                model=thieved_model._model,
                loss=thieved_model.criterion,
                input_shape=dataset.data_shape,
                nb_classes=thieved_model.num_classes,
                optimizer=trainer.optimizer
            )
            thieved_model._validate(print_prefix='Before Stealing')
            attack = AttackClass(classifier, batch_size_fit=dataset.batch_size, batch_size_query=dataset.batch_size,
                                 nb_epochs=20, nb_stolen=50000, use_probability=use_probability)
            attack.extract(x=x_train, y=y_train, thieved_classifier=thieved_classifier)

            thieved_model.activate_params([])
            thieved_model.eval()
            thieved_model._validate(print_prefix='After Stealing')
            print('-' * 20)
            print('\n\n')
