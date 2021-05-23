#!/usr/bin/env python3

import trojanvision
from trojanvision.utils import summary, to_numpy
from trojanzoo.utils.data import dataset_to_list

import torch
import torch.nn as nn
import argparse


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--nb_stolen', type=int, dest="nb_stolen", default=5000)
    parser.add_argument('--tmodel', dest="tmodel", default=None)
    parser.add_argument('--tmodel_arch', dest="tmodel_arch", default=None)
    trojanvision.environ.add_argument(parser)
    trojanvision.datasets.add_argument(parser)
    trojanvision.models.add_argument(parser)
    trojanvision.trainer.add_argument(parser)
    args = parser.parse_args()
    nb_stolen = args.nb_stolen
    # print(nb_stolen)

    env = trojanvision.environ.create(**args.__dict__)
    dataset = trojanvision.datasets.create(**args.__dict__)
    model = trojanvision.models.create(dataset=dataset, **args.__dict__)

    if env['verbose']:
        summary(env=env, dataset=dataset, model=model)
    # model._validate()
    model.eval()
    # print('\n\n')

    from art.estimators.classification import PyTorchClassifier  # type: ignore
    classifier = PyTorchClassifier(
        model=model._model,
        loss=model.criterion,
        input_shape=dataset.data_shape,
        nb_classes=model.num_classes,
    )
    x_train, y_train = dataset_to_list(dataset.get_dataset('train'))
    x_train, y_train = to_numpy(torch.stack(x_train)), to_numpy(y_train)

    import art.attacks.extraction  # type:ignore
    # for name in ['CopycatCNN', 'KnockoffNets']:
    AttackClass = getattr(art.attacks.extraction, 'KnockoffNets')
    for mode in ['random']:

        model_name = args.model_name if not args.tmodel else args.tmodel

        # print(model_name, model_arch)

        args_dict = args.__dict__ | {'pretrain': False, 'official': False,
                                     'model_name': model_name}
        if args.tmodel_arch is not None:
            args_dict['model_arch'] = args.tmodel_arch
        elif 'model_arch' in args.__dict__.keys():
            args_dict['model_arch'] = args.model_arch

        thieved_model = trojanvision.models.create(dataset=dataset, **args_dict)
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

        # thieved_model._validate(print_prefix='Before Stealing')
        attack = AttackClass(classifier, batch_size_fit=dataset.batch_size, batch_size_query=dataset.batch_size,
                             nb_epochs=25, nb_stolen=nb_stolen, use_probability=True, sampling_strategy=mode, verbose=False)
        attack.extract(x=x_train, y=y_train, thieved_classifier=thieved_classifier)

        thieved_model.activate_params([])
        thieved_model.eval()
        thieved_model._validate(print_prefix='KnockoffNets-' + mode)
        model._compare(thieved_model)
        print('-' * 20)
        # print('\n\n')
