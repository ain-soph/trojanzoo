#!/usr/bin/env python3

# Normal Train
# CUDA_VISIBLE_DEVICES=0 python main.py --dataset mnist --epoch_eval_train 300 --image_per_class 1
# CUDA_VISIBLE_DEVICES=0 python main.py --dataset cifar10 --epoch_eval_train 100 --image_per_class 10

# ResNet Model
# --model resnet18_ap_comp --eval_model resnet18_comp --eval_norm_layer gn --dataset_normalize

# FGSM adv train
# CUDA_VISIBLE_DEVICES=0 python main.py --dataset mnist --epoch_eval_train 300 --adv_train --eval_adv_train --adv_train_iter 1 --adv_train_alpha 0.375 --first_term fgsm --image_per_class 1
# CUDA_VISIBLE_DEVICES=0 python main.py --dataset cifar10 --epoch_eval_train 100 --adv_train --eval_adv_train --adv_train_iter 1 --adv_train_alpha 0.0392156862745 --first_term fgsm --image_per_class 10

# PGD adv train
# CUDA_VISIBLE_DEVICES=0 python main.py --dataset mnist --epoch_eval_train 300 --adv_train --eval_adv_train --first_term fgsm --dis_metric kfac --image_per_class 1
# CUDA_VISIBLE_DEVICES=0 python main.py --dataset cifar10 --epoch_eval_train 100 --adv_train --eval_adv_train --first_term fgsm --dis_metric kfac --image_per_class 10

# https://github.com/VICO-UoE/DatasetCondensation

from projects.dataset_condensation.utils import freeze_bn
import trojanvision
from trojanvision.utils import summary
import argparse

from trojanvision.utils.model import weight_init
from trojanzoo.utils.tensor import save_tensor_as_img
from trojanzoo.utils.logger import SmoothedValue
from trojanzoo.utils.output import prints, ansi
from trojanzoo.utils.fim import fim, fim_diag, KFAC

import torch
from torch.utils.data import TensorDataset
from torchvision import transforms
import functools

from utils import match_loss, augment, get_daparam, get_loops
from model import ConvNet

trojanvision.models.class_dict['convnet'] = ConvNet


default_args = {
    'verbose': 1,
    'color': True,
    'lr': 0.01,
    'momentum': 0.0,
    'weight_decay': 0.0,
    'validate_interval': 0,
    'batch_size': 256,
    'model_name': 'convnet',
    'adv_train_random_init': True,
}

eval_args = {
    'momentum': 0.9,
    'weight_decay': 5e-4,
    'adv_train': True,
}

fgsm_args = {
    'cifar10': {
        'iteration': 1,
        'pgd_alpha': 10.0 / 255,
        'pgd_eps': 8.0 / 255,
    },
    'mnist': {
        'iteration': 1,
        'pgd_alpha': 0.375,
        'pgd_eps': 0.3,
    },
}

pgd_args = {
    'cifar10': {
        'iteration': 7,
        'pgd_alpha': 2.0 / 255,
        'pgd_eps': 8.0 / 255,
    },
    'mnist': {
        'iteration': 7,
        'pgd_alpha': 0.1,
        'pgd_eps': 0.3,
    },
}

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
    parser.add_argument('--file_path')
    parser.add_argument('--init', default='noise', choices=['noise', 'real'],
                        help='noise/real: initialize synthetic images from random noise or randomly sampled real images.')
    parser.add_argument('--first_term', choices=['fgsm', 'pgd'])
    parser.add_argument('--eval_adv_train', action='store_true')
    args = parser.parse_args()

    dis_metric: str = args.dis_metric
    image_per_class: int = args.image_per_class
    num_eval: int = args.num_eval
    epoch_eval_train: int = args.epoch_eval_train
    Iteration: int = args.Iteration
    lr_img: float = args.lr_img
    outer_loop, inner_loop = get_loops(image_per_class)

    args_dict = args.__dict__
    args_dict.update(default_args)
    env = trojanvision.environ.create(**args_dict)
    dataset = trojanvision.datasets.create(**args_dict)

    fgsm_dict = fgsm_args[dataset.name]
    pgd_dict = pgd_args[dataset.name]
    args_dict['adv_train_eval_iter'] = pgd_dict['iteration']
    args_dict['adv_train_eval_alpha'] = pgd_dict['pgd_alpha']
    args_dict['adv_train_eval_eps'] = pgd_dict['pgd_eps']
    model = trojanvision.models.create(dataset=dataset, **args_dict)
    trainer = trojanvision.trainer.create(dataset=dataset, model=model, **args_dict)

    if env['verbose']:
        summary(env=env, dataset=dataset, model=model, trainer=trainer)

    if args_dict['eval_model'] is not None:
        args_dict['model_name'] = args_dict['eval_model']
    if args_dict['eval_norm_layer'] is not None:
        args_dict['norm_layer'] = args_dict['eval_norm_layer']
    args_dict.update(eval_args)
    if args.first_term == 'fgsm':
        args_dict['adv_train_iter'] = fgsm_dict['iteration']
        args_dict['adv_train_alpha'] = fgsm_dict['pgd_alpha']
        args_dict['adv_train_eps'] = fgsm_dict['pgd_eps']
    if args.first_term == 'pgd':
        args_dict['adv_train_iter'] = pgd_dict['iteration']
        args_dict['adv_train_alpha'] = pgd_dict['pgd_alpha']
        args_dict['adv_train_eps'] = pgd_dict['pgd_eps']
    eval_model = trojanvision.models.create(dataset=dataset, **args_dict)
    eval_trainer = trojanvision.trainer.create(dataset=dataset, model=eval_model, **args_dict)

    if env['verbose']:
        summary(eval_model=eval_model, eval_trainer=eval_trainer)

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
    # train_args['epoch'] = 1
    train_args['epoch'] = inner_loop
    train_args['lr_scheduler'] = None
    train_args['adv_train'] = False if args.first_term is None else True
    eval_train_args = dict(**eval_trainer)
    eval_train_args['epoch'] = epoch_eval_train
    eval_train_args['adv_train'] = args.eval_adv_train
    mean_value = [0.0]
    std_value = [1.0]
    if dataset.norm_par is not None:
        mean_value = dataset.norm_par['mean']
        std_value = dataset.norm_par['std']
    mean = torch.tensor(mean_value, device=env['device'])[:, None, None]
    std = torch.tensor(std_value, device=env['device'])[:, None, None]

    param_augment = get_daparam(dataset.name)
    net_parameters = list(model.parameters())

    kfac = None
    if dis_metric == 'kfac':
        kfac = KFAC(model)

    def get_data_fn(data, **kwargs):
        _input, _label = eval_model.get_data(data, **kwargs)
        _input = augment(_input, param_augment, device=env['device'])
        return _input, _label

    def get_real_grad(img_real: torch.Tensor, lab_real: torch.Tensor,
                      adv_train: bool = False) -> list[torch.Tensor]:
        if adv_train:
            img_real, _ = model.pgd.optimize(_input=img_real, target=lab_real,
                                             iteration=model.adv_train_iter,
                                             pgd_alpha=model.adv_train_alpha,
                                             pgd_eps=model.adv_train_eps)

        if dis_metric == 'kfac':
            kfac.track.enable()
        loss_real = model.loss(img_real, lab_real)
        gw_real = list((grad.detach().clone() for grad in torch.autograd.grad(loss_real, net_parameters)))
        if dis_metric == 'kfac':
            kfac.track.disable()
            kfac.update_covs()
            kfac.update_inv_covs()
        return gw_real

    def get_syn_grad(img_syn: torch.Tensor, lab_syn: torch.Tensor) -> list[torch.Tensor]:
        loss_syn = model.loss(img_syn, lab_syn)
        gw_syn = torch.autograd.grad(loss_syn, net_parameters, create_graph=True)
        return gw_syn

    def get_real_data(c: int, batch_size: int, **kwargs) -> torch.Tensor:
        try:
            data_real = next(iter_list[c])
        except StopIteration as e:
            iter_list[c] = iter(class_loader_list[c])   # don't use itertools.cycle because it's not random
            data_real = next(iter_list[c])
        return model.get_data(data_real)[0][:batch_size]

    # images_all, labels_all = dataset_to_list(train_set)
    # images_all = torch.stack(images_all).to(env['device'])
    # labels_all = torch.tensor(labels_all, dtype=torch.long, device=env['device'])
    # indices_class = [[] for c in range(dataset.num_classes)]
    # for i, lab in enumerate(labels_all):
    #     indices_class[lab].append(i)
    #
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

    if args.init == 'real':
        bn_size = image_syn.shape[1]
        for c in range(dataset.num_classes):
            image_syn[c].data = get_real_data(c, bn_size)
    best_result = 0.0
    for it in range(Iteration):
        ''' Evaluate synthetic data '''
        # print(f'    {it+1:4d}')
        if (it + 1) % args.eval_interval == 0:
            if outer_loop > 1:
                model._validate()
            accs = SmoothedValue(fmt='{global_avg:7.3f} ({min:7.3f}  {max:7.3f})')
            robusts = SmoothedValue(fmt='{global_avg:7.3f} ({min:7.3f}  {max:7.3f})')
            for _ in range(num_eval):
                # eval_model.load(suffix='')
                weight_init(eval_model._model)
                dst_syn_train = TensorDataset(image_syn.detach().clone().flatten(0, 1),
                                              label_syn.detach().clone().flatten(0, 1))
                loader_train = dataset.get_dataloader(mode='train', pin_memory=False, num_workers=0,
                                                      dataset=dst_syn_train)
                eval_model._train(loader_train=loader_train, verbose=False, get_data_fn=get_data_fn,
                                  **eval_train_args)
                result_a, result_b = eval_model._validate(adv_train=model.adv_train, verbose=False)
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
            print(f'{it+1:4d}    acc: ', accs, end='    ')
            if model.adv_train:
                print('robust: ', robusts, end='    ')
            print(f'images statistics: mean: {float(image_syn.mean()):.2f}  '
                  f'median: {float(image_syn.median()):.2f} '
                  f'({float(image_syn.min()):.2f}, {float(image_syn.max()):.2f})')
            cur_result = robusts.global_avg + accs.global_avg if eval_model.adv_train else accs.global_avg
            if cur_result > best_result:
                best_result = cur_result
                print(' ' * 12, '{purple}best result update!{reset}'.format(**ansi))
                if args.file_path:
                    data = {'image_syn': image_syn, 'label_syn': label_syn}
                    torch.save(data, args.file_path)
                    print(' ' * 12, 'file save at: ', args.file_path)
                if args.save_img:
                    for c in range(len(image_syn)):
                        for i in range(len(image_syn[c])):
                            filename = f'./result/dataset_condensation/{dataset.name}_{c}_{i}.png'
                            image = image_syn[c][i]
                            if dataset.normalize:
                                image = image.add(mean).mul(std)
                            image = image.clamp(0, 1)
                            save_tensor_as_img(filename, image)
        # model.load(suffix='')
        weight_init(model._model)

        # warmup_args = dict(**train_args)
        # warmup_args['epoch'] = 3
        # warmup_args['lr_scheduler'] = None
        # warmup_args['adv_train'] = False
        # model._train(verbose=False, change_train_eval=False, **warmup_args)

        # loss_avg = 0.0
        for ol in range(outer_loop):
            model.activate_params(net_parameters)
            model.train()

            freeze_bn(model, get_real_data)

            fim_inv_list: list[torch.Tensor] = None
            if dis_metric == 'natural':
                fim_list = fim_diag(model._model, image_syn.detach().clone().flatten(0, 1))
                fim_inv_list = [fim.detach() for fim in fim_list]
                fim_inv_list = [fim.add(1e-6).reciprocal() for fim in fim_list]   # 1/fim
                fim_inv_list = [fim / (fim.abs().max() + 1e-6) for fim in fim_list]
            elif dis_metric == 'natural_full':
                fim_list = fim(model._model, image_syn.detach().clone().flatten(0, 1))
                fim_inv_list = [fim.detach() for fim in fim_list]
                fim_inv_list = [fim.add(1e-6 * torch.eye(fim.shape[0], dtype=fim.device)).inverse()
                                for fim in fim_list]
                fim_inv_list = [(fim + fim.t()) / 2 for fim in fim_list]
                fim_inv_list = [fim / (fim.abs().max() + 1e-6) for fim in fim_list]

            # for _ in range(10):
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

                if args.first_term is not None:
                    perturbed, _ = model.pgd.optimize(_input=img_syn.detach(), target=lab_syn,
                                                      **fgsm_args[dataset.name])
                    img_syn = img_syn + (perturbed - img_syn).detach()
                gw_syn = get_syn_grad(img_syn, lab_syn)
                if model.adv_train_free:
                    gw_real = get_real_grad(img_real, lab_real, adv_train=False)
                    loss += match_loss(gw_syn, gw_real, dis_metric, fim_inv_list=fim_inv_list, kfac=kfac)
                gw_real = get_real_grad(img_real, lab_real, adv_train=model.adv_train)
                loss += match_loss(gw_syn, gw_real, dis_metric, fim_inv_list=fim_inv_list, kfac=kfac)
            optimizer_img.zero_grad()
            loss.backward()
            # image_syn.grad.data.sign_()
            optimizer_img.step()
            # loss_avg += loss.item()
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

            # sub_loader = dataset.get_dataloader(mode='train', full=False, seed=random.randint(0, 1000))
            # model._train(verbose=False, loader_train=sub_loader, **train_args)
            ''' update network '''
            dst_syn_train = TensorDataset(image_syn.detach().clone().flatten(0, 1),
                                          label_syn.detach().clone().flatten(0, 1))
            loader_train = dataset.get_dataloader(mode='train', num_workers=0, pin_memory=False, dataset=dst_syn_train)
            model._train(loader_train=loader_train, verbose=False, change_train_eval=False, **train_args)
        # loss_avg /= (model.num_classes * outer_loop)
        # if it % 10 == 0:
        # print('iter = %04d, loss = %.4f' % (it, loss_avg))
