#!/usr/bin/env python3

# CUDA_VISIBLE_DEVICES=0 python main.py --verbose 1 --color --num_workers 0 --lr 0.01 --momentum 0.0 --weight_decay 0.0 --validate_interval 0 --dataset mnist --batch_size 256 --epoch 300 --model convnet
# CUDA_VISIBLE_DEVICES=0 python main.py --verbose 1 --color --num_workers 0 --lr 0.01 --momentum 0.0 --weight_decay 0.0 --validate_interval 0 --dataset mnist --batch_size 256 --epoch 300 --adv_train --adv_train_random_init --adv_train_iter 1 --adv_train_alpha 0.375 --adv_train_eps 0.3 --adv_train_eval_iter 7 --adv_train_eval_alpha 0.0078431372549 --adv_train_eval_eps 0.031372549 --model convnet


# CUDA_VISIBLE_DEVICES=0 python main.py --verbose 1 --color --num_workers 0 --lr 0.01 --momentum 0.0 --weight_decay 0.0 --validate_interval 0 --dataset cifar10 --batch_size 256 --epoch 300 --epoch_eval_train 300 --image_per_class 10 --model resnet18_ap_comp --eval_model resnet18_comp --eval_norm_layer gn
# CUDA_VISIBLE_DEVICES=0 python main.py --verbose 1 --color --num_workers 0 --lr 0.01 --momentum 0.0 --weight_decay 0.0 --validate_interval 0 --dataset cifar10 --batch_size 256 --epoch 300 --epoch_eval_train 300 --adv_train --adv_train_random_init --adv_train_iter 1 --adv_train_alpha 0.0392156862745 --adv_train_eval_iter 7 --adv_train_eval_alpha 0.0078431372549 --image_per_class 10 --model resnet18_ap_comp --eval_model resnet18_comp --eval_norm_layer gn

# --model resnet18_ap_comp --eval_model resnet18_comp --eval_norm_layer gn
#

# https://github.com/VICO-UoE/DatasetCondensation

import trojanvision
from trojanvision.utils import summary
import argparse

from trojanvision.utils.model import weight_init
from trojanzoo.utils.tensor import save_tensor_as_img
from trojanzoo.utils.logger import SmoothedValue
from trojanzoo.utils.output import prints, ansi
from trojanzoo.utils.fim import fim_diag
from trojanzoo.utils.data import dataset_to_list

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset
from torchvision import transforms
import functools

from utils import match_loss, augment, get_daparam
from model import ConvNet

trojanvision.models.class_dict['convnet'] = ConvNet

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    trojanvision.environ.add_argument(parser)
    trojanvision.datasets.add_argument(parser)
    trojanvision.models.add_argument(parser)
    trojanvision.trainer.add_argument(parser)
    parser.add_argument('--eval_model')
    parser.add_argument('--eval_norm_layer')
    parser.add_argument('--dis_metric', default='ours')
    parser.add_argument('--image_per_class', type=int, default=1)
    parser.add_argument('--num_eval', type=int, default=3)
    parser.add_argument('--epoch_eval_train', type=int, default=1000)
    parser.add_argument('--Iteration', type=int, default=1000)
    parser.add_argument('--lr_img', type=float, default=0.1)
    parser.add_argument('--save_img', action='store_true')
    parser.add_argument('--eval_interval', type=int, default=10)
    args = parser.parse_args()

    dis_metric: str = args.dis_metric
    image_per_class: int = args.image_per_class
    num_eval: int = args.num_eval
    epoch_eval_train: int = args.epoch_eval_train
    Iteration: int = args.Iteration
    lr_img: float = args.lr_img
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
    args_dict = args.__dict__
    env = trojanvision.environ.create(**args_dict)
    dataset = trojanvision.datasets.create(**args_dict)
    model = trojanvision.models.create(dataset=dataset, **args_dict)
    trainer = trojanvision.trainer.create(dataset=dataset, model=model, **args_dict)

    if env['verbose']:
        summary(env=env, dataset=dataset, model=model, trainer=trainer)

    if args_dict['eval_model'] is not None:
        args_dict['model_name'] = args_dict['eval_model']
    if args_dict['eval_norm_layer'] is not None:
        args_dict['norm_layer'] = args_dict['eval_norm_layer']
    args_dict['momentum'] = 0.9
    args_dict['weight_decay'] = 5e-4
    eval_model = trojanvision.models.create(dataset=dataset, **args_dict)
    eval_trainer = trojanvision.trainer.create(dataset=dataset, model=eval_model, **args_dict)

    init_fn = torch.randn if dataset.normalize else torch.rand
    image_syn = init_fn(size=(model.num_classes, image_per_class, *dataset.data_shape),
                        dtype=torch.float, requires_grad=True, device=env['device'])
    label_syn = torch.stack([c * torch.ones(image_per_class, dtype=torch.long, device=env['device'])
                             for c in range(model.num_classes)])
    image_syn.requires_grad_()
    optimizer_img = torch.optim.SGD([image_syn, ], lr=lr_img, momentum=0.5)  # optimizer_img for synthetic data
    optimizer_img.zero_grad()
    # lr_scheduler_img = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_img, T_max=Iteration * outer_loop)

    transform = [transforms.ToTensor()]
    if dataset.normalize and dataset.norm_par is not None:
        transform.append(transforms.Normalize(mean=dataset.norm_par['mean'], std=dataset.norm_par['std']))
    train_set = dataset.get_full_dataset(mode='train', transform=transforms.Compose(transform))
    # class_list = [torch.stack(dataset_to_list(dataset.get_dataset(dataset=train_set, class_list=[c]))[0])
    #               for c in range(model.num_classes)]
    class_loader_list = [dataset.get_dataloader(mode='train', drop_last=True, num_workers=0, pin_memory=False,
                                                dataset=dataset.get_dataset(dataset=train_set, class_list=[c]))
                         for c in range(model.num_classes)]
    iter_list = [iter(loader) for loader in class_loader_list]

    train_args = dict(**trainer)
    train_args['epoch'] = inner_loop
    train_args['lr_scheduler'] = None
    train_args['adv_train'] = False
    eval_train_args = dict(**eval_trainer)
    eval_train_args['epoch'] = epoch_eval_train
    eval_train_args['adv_train'] = False
    if dataset.norm_par is not None:
        mean_value = dataset.norm_par['mean']
        std_value = dataset.norm_par['std']
    else:
        mean_value = [0.0]
        std_value = [1.0]
    mean = torch.tensor(mean_value, device=env['device'])[:, None, None]
    std = torch.tensor(std_value, device=env['device'])[:, None, None]

    param_augment = get_daparam(dataset.name)
    net_parameters = list(model.parameters())

    def get_data_fn(data, **kwargs):
        _input, _label = model.get_data(data, **kwargs)
        _input = augment(_input, param_augment, device=env['device'])
        return _input, _label

    def get_real_grad(img_real: torch.Tensor, lab_real: torch.Tensor,
                      adv_train: bool = False) -> list[torch.Tensor]:
        if adv_train:
            adv_loss_fn = functools.partial(model._adv_loss_helper, _label=lab_real)
            img_real, _ = model.pgd.optimize(_input=img_real, loss_fn=adv_loss_fn,
                                             iteration=model.adv_train_iter,
                                             pgd_alpha=model.adv_train_alpha,
                                             pgd_eps=model.adv_train_eps)
        loss_real = model.loss(img_real, lab_real)
        gw_real = list((grad.detach().clone() for grad in torch.autograd.grad(loss_real, net_parameters)))
        return gw_real

    def get_syn_grad(img_syn: torch.Tensor, lab_syn: torch.Tensor) -> list[torch.Tensor]:
        loss_syn = model.loss(img_syn, lab_syn)
        gw_syn = torch.autograd.grad(loss_syn, net_parameters, create_graph=True)
        return gw_syn

    images_all, labels_all = dataset_to_list(train_set)
    images_all = torch.stack(images_all).to(env['device'])
    labels_all = torch.tensor(labels_all, dtype=torch.long, device=env['device'])
    indices_class = [[] for c in range(dataset.num_classes)]
    for i, lab in enumerate(labels_all):
        indices_class[lab].append(i)

    def get_real_data(c: int, *args, **kwargs) -> torch.Tensor:
        try:
            data_real = next(iter_list[c])
        except StopIteration as e:
            iter_list[c] = iter(class_loader_list[c])   # don't use itertools.cycle because it's not random
            data_real = next(iter_list[c])
        return model.get_data(data_real)[0]

    # def get_real_data(c: int, n: int) -> torch.Tensor:
    #     idx_shuffle = np.random.permutation(indices_class[c])[:n]
    #     return images_all[idx_shuffle].detach().clone()

    # def get_real_data(c: int, n: int) -> torch.Tensor:
    #     idx_shuffle = torch.randperm(len(class_list[c]))[:n]
    #     return class_list[c][idx_shuffle]

    # def get_real_data(c: int, n: int) -> torch.Tensor:
    #     idx_shuffle = torch.randperm(len(class_list[c]))[:n]
    #     return class_list[c][idx_shuffle].to(env['device'])

    # from data import get_images
    # get_real_data = get_images

    for it in range(Iteration):
        ''' Evaluate synthetic data '''
        # print(f'    {it+1:4d}')
        if it % args.eval_interval == 0:
            accs = SmoothedValue(fmt='{global_avg:7.3f} ({min:7.3f}  {max:7.3f})')
            robusts = SmoothedValue(fmt='{global_avg:7.3f} ({min:7.3f}  {max:7.3f})')
            for _ in range(num_eval):
                weight_init(eval_model._model)
                dst_syn_train = TensorDataset(image_syn.detach().clone().flatten(0, 1),
                                              label_syn.detach().clone().flatten(0, 1))
                loader_train = dataset.get_dataloader(mode='train', pin_memory=False, num_workers=0,
                                                      dataset=dst_syn_train)
                eval_model._train(loader_train=loader_train, verbose=False, get_data_fn=get_data_fn, **eval_train_args)
                result_a, result_b = eval_model._validate(verbose=False)
                if model.adv_train:
                    acc, robust = result_b - result_a, result_a
                    accs.update(acc)
                    robusts.update(robust)
                    prints('{green}acc: {yellow}{0:7.3f}  {green}robust: {yellow}{1:7.3f}{reset}'.format(
                        acc, robust, **ansi), indent=20)
                else:
                    accs.update(result_b)
                    prints('{green}acc: {yellow}{0:7.3f}{reset}'.format(
                        result_b, **ansi), indent=20)
            print(f'{it:4d}    acc: ', accs, end='    ')
            if model.adv_train:
                print('robust: ', robusts, end='    ')
            print(
                f'images statistics: mean: {float(image_syn.mean()):.2f}  median: {float(image_syn.median()):.2f} ({float(image_syn.min()):.2f}, {float(image_syn.max()):.2f})')
            if args.save_img:
                for c in range(len(image_syn)):
                    for i in range(len(image_syn[c])):
                        filename = f'./result/dataset_condensation/{dataset.name}_{c}_{i}.png'
                        image = image_syn[c][i]
                        if dataset.normalize:
                            image = image.add(mean).mul(std)
                        image = image.clamp(0, 1)
                        save_tensor_as_img(filename, image)

        weight_init(model._model)

        loss_avg = 0.0
        for ol in range(outer_loop):
            model.activate_params(net_parameters)
            model.train()

            # freeze the running mu and sigma for BatchNorm layers
            BN_flag = False
            BNSizePC = 16  # for batch normalization
            for module in model.modules():
                if 'BatchNorm' in module._get_name():  # BatchNorm
                    BN_flag = True
            if BN_flag:
                img_real_list: list[torch.Tensor] = []
                with torch.no_grad():
                    img_real = torch.cat([get_real_data(c, BNSizePC) for c in range(model.num_classes)], dim=0)
                    output_real = model(img_real)  # get running mu, sigma
                for module in model.modules():
                    if isinstance(module, nn.BatchNorm2d):
                        module.eval()

            fim_inv_list: list[torch.Tensor] = None
            if dis_metric == 'natural':
                fim_list = fim_diag(model._model, image_syn.detach().clone().flatten(0, 1))
                fim_inv_list = [fim.add(1e-6).reciprocal() for fim in fim_list]   # 1/fim
                # fim_inv_list = [torch.ones_like(fim) for fim in fim_list]   # 1/fim
                fim_inv_list = [fim / (fim.abs().max() + 1e-6) for fim in fim_list]   # 1/fim

            model.zero_grad()
            loss = torch.tensor(0.0, device=env['device'])
            for c in range(model.num_classes):
                # try:
                #     data_real = next(iter_list[c])
                # except StopIteration as e:
                #     iter_list[c] = iter(class_loader_list[c])   # don't use itertools.cycle because it's not random
                #     data_real = next(iter_list[c])
                # img_real, lab_real = model.get_data(data_real)
                img_real = get_real_data(c, dataset.batch_size)
                lab_real = c * torch.ones(img_real.size(0), device=img_real.device, dtype=torch.long)

                img_syn = image_syn[c]
                lab_syn = label_syn[c]

                gw_syn = get_syn_grad(img_syn, lab_syn)
                if model.adv_train_free:
                    gw_real = get_real_grad(img_real, lab_real, adv_train=False)
                    loss += match_loss(gw_syn, gw_real, dis_metric, fim_inv_list=fim_inv_list)
                gw_real = get_real_grad(img_real, lab_real, adv_train=model.adv_train)
                loss += match_loss(gw_syn, gw_real, dis_metric, fim_inv_list=fim_inv_list)
            optimizer_img.zero_grad()
            loss.backward()
            optimizer_img.step()
            loss_avg += loss.item()
            # lr_scheduler_img.step()
            # for c in range(dataset.data_shape[0]):
            #     mean = dataset.norm_par['mean'][c]
            #     std = dataset.norm_par['std'][c]
            #     image_syn.data[c] = image_syn[c].clamp(-mean / std, (1 - mean) / std)
            # mean = dataset.norm_par['mean'][0]
            # std = dataset.norm_par['std'][0]
            # image_syn.data.clamp_(-mean / std, (1 - mean) / std)
            # image_syn.data.clamp_(-0.1, 1.1)

            if ol == outer_loop - 1:
                break

            ''' update network '''
            dst_syn_train = TensorDataset(image_syn.detach().clone().flatten(0, 1),
                                          label_syn.detach().clone().flatten(0, 1))
            loader_train = dataset.get_dataloader(mode='train', num_workers=0, pin_memory=False, dataset=dst_syn_train)
            model._train(loader_train=loader_train, verbose=False, change_train_eval=False, **train_args)
        loss_avg /= (model.num_classes * outer_loop)
        # if it % 10 == 0:
        print('iter = %04d, loss = %.4f' % (it, loss_avg))
