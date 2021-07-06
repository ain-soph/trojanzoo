#!/usr/bin/env python3

# CUDA_VISIBLE_DEVICES=0 python projects/dataset_condensation/main.py --verbose 1 --color --lr 0.01 --validate_interval 0 --dataset mnist --batch_size 256 --weight_decay 5e-4 --lr_scheduler --epoch 300 --dataset_normalize

# https://github.com/VICO-UoE/DatasetCondensation

import trojanvision
from trojanvision.utils import summary
import argparse

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset
from trojanvision.utils.model import weight_init
from trojanzoo.utils.tensor import save_tensor_as_img
from trojanzoo.utils.logger import SmoothedValue

from utils import match_loss, augment, get_daparam
from model import ConvNet

Iteration = 1000
image_per_class = 1
dis_metric = 'ours'
lr_img = 0.1

num_eval = 5
epoch_eval_train = 1000


if image_per_class == 1:
    outer_loop, inner_loop = 1, 1
elif image_per_class == 10:
    outer_loop, inner_loop = 10, 50
elif image_per_class == 20:
    outer_loop, inner_loop = 20, 25
elif image_per_class == 30:
    outer_loop, inner_loop = 30, 20
elif image_per_class == 40:
    outer_loop, inner_loop = 40, 15
elif image_per_class == 50:
    outer_loop, inner_loop = 50, 10
else:
    outer_loop, inner_loop = 0, 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    trojanvision.environ.add_argument(parser)
    trojanvision.datasets.add_argument(parser)
    trojanvision.models.add_argument(parser)
    trojanvision.trainer.add_argument(parser)
    args = parser.parse_args()

    trojanvision.models.class_dict['convnet'] = ConvNet

    env = trojanvision.environ.create(**args.__dict__)
    dataset = trojanvision.datasets.create(**args.__dict__)
    model = trojanvision.models.create(dataset=dataset, **args.__dict__)
    trainer = trojanvision.trainer.create(dataset=dataset, model=model, **args.__dict__)

    if env['verbose']:
        summary(env=env, dataset=dataset, model=model, trainer=trainer)
    eval_model = trojanvision.models.create(dataset=dataset, **args.__dict__)
    eval_trainer = trojanvision.trainer.create(dataset=dataset, model=eval_model, **args.__dict__)

    image_syn = torch.randn(model.num_classes, image_per_class, *dataset.data_shape, device=env['device'])
    label_syn = torch.stack([c * torch.ones(image_per_class, dtype=torch.long, device=env['device'])
                             for c in range(model.num_classes)])
    image_syn.requires_grad_()
    optimizer_img = torch.optim.SGD([image_syn, ], lr=lr_img, momentum=0.5)  # optimizer_img for synthetic data
    optimizer_img.zero_grad()
    # lr_scheduler_img = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_img, T_max=Iteration * outer_loop)

    train_set = dataset.get_full_dataset(mode='train')
    class_loader_list = [dataset.get_dataloader(mode='train', drop_last=True, num_workers=0,
                                                dataset=dataset.get_dataset(dataset=train_set, class_list=[c]))
                         for c in range(model.num_classes)]
    iter_list = [iter(loader) for loader in class_loader_list]
    train_args = dict(**trainer)
    train_args['epoch'] = inner_loop
    train_args['optimizer'] = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.5)
    train_args['lr_scheduler'] = None
    eval_train_args = dict(**eval_trainer)
    eval_train_args['epoch'] = epoch_eval_train // 2
    mean = torch.tensor(dataset.norm_par['mean'], device=env['device'])[:, None, None]
    std = torch.tensor(dataset.norm_par['std'], device=env['device'])[:, None, None]

    param_augment = get_daparam(dataset.name)

    def get_data_fn(data, **kwargs):
        _input, _label = dataset.get_data(data, **kwargs)
        _input = augment(_input, param_augment, device=env['device'])
        return _input, _label

    # freeze the running mu and sigma for BatchNorm layers
    # model.load()
    # weight_init(model._model, filter_list=(nn.BatchNorm2d, ))
    model.train()
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.eval()
    for it in range(Iteration):
        ''' Evaluate synthetic data '''
        # print(f'    {it+1:4d}')
        if (it + 1) % 10 == 0 or it == 0:
            accs = SmoothedValue(fmt='{global_avg:.3f} ({min:.3f}  {max:.3f})')
            for _ in range(num_eval):
                weight_init(eval_model._model)
                dst_syn_train = TensorDataset(image_syn.detach().clone().flatten(0, 1),
                                              label_syn.detach().clone().flatten(0, 1))
                loader_train = dataset.get_dataloader(mode='train', pin_memory=False, num_workers=0,
                                                      dataset=dst_syn_train)
                eval_model._train(loader_train=loader_train, verbose=False, get_data_fn=get_data_fn, **eval_train_args)
                eval_train_args['optimizer'].param_groups[0]['lr'] *= 0.1
                eval_model._train(loader_train=loader_train, verbose=False, get_data_fn=get_data_fn, **eval_train_args)
                _, acc = eval_model._validate(verbose=False)
                accs.update(acc)
            print(f'{it+1:4d}: ', accs, end='    ')
            print(
                f'images statistics: mean: {float(image_syn.mean()):.2f}  median: {float(image_syn.median()):.2f} ({float(image_syn.min()):.2f}, {float(image_syn.max()):.2f})')
            for c in range(len(image_syn)):
                for i in range(len(image_syn[c])):
                    filename = f'./result/dataset_condensation/{dataset.name}_{c}_{i}.png'
                    image = image_syn[c][i]
                    if dataset.normalize:
                        image = image.add(mean).mul(std)
                    image = image.clamp(0, 1)
                    save_tensor_as_img(filename, image)

        weight_init(model._model, filter_list=(nn.BatchNorm2d, ))
        net_parameters = list(model.parameters())
        for ol in range(outer_loop):
            model.activate_params(net_parameters)
            loss = torch.tensor(0.0, device=env['device'])
            for c in range(model.num_classes):
                try:
                    data_real = next(iter_list[c])
                except StopIteration as e:
                    iter_list[c] = iter(class_loader_list[c])   # don't use itertools.cycle because it's not random
                    data_real = next(iter_list[c])
                img_real, lab_real = model.get_data(data_real)
                loss_real = model.loss(img_real, lab_real)
                gw_real = list((grad.detach().clone() for grad in torch.autograd.grad(loss_real, net_parameters)))

                img_syn = image_syn[c]
                lab_syn = label_syn[c]
                loss_syn = model.loss(img_syn, lab_syn)
                gw_syn = torch.autograd.grad(loss_syn, net_parameters, create_graph=True)

                loss += match_loss(gw_syn, gw_real, dis_metric)
            optimizer_img.zero_grad()
            loss.backward()
            optimizer_img.step()
            # lr_scheduler_img.step()
            # for c in range(dataset.data_shape[0]):
            #     mean = dataset.norm_par['mean'][c]
            #     std = dataset.norm_par['std'][c]
            #     image_syn.data[c] = image_syn[c].clamp(-mean / std, (1 - mean) / std)
            # mean = dataset.norm_par['mean'][0]
            # std = dataset.norm_par['std'][0]
            # image_syn.data.clamp_(-mean / std, (1 - mean) / std)

            if ol == outer_loop - 1:
                break

            ''' update network '''
            dst_syn_train = TensorDataset(image_syn.detach().clone().flatten(0, 1),
                                          label_syn.detach().clone().flatten(0, 1))
            loader_train = dataset.get_dataloader(mode='train', num_workers=0, pin_memory=False, dataset=dst_syn_train)
            model._train(loader_train=loader_train, get_data_fn=get_data_fn,
                         verbose=False, change_train_eval=False, **train_args)
