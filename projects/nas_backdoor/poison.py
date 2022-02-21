#!/usr/bin/env python3

# CUDA_VISIBLE_DEVICES=0 python projects/nas_backdoor/poison.py --verbose 1 --color --attack poison_basic --dataset cifar10 --model darts --supernet --layers 8 --init_channels 16 --pretrained --validate_interval 1 --target_idx -1 --epoch 5 --clean_epoch 2 --only_paramless_op
# --arch_unrolled

import trojanvision
import argparse


from trojanvision.utils.model_archs.darts.operations import PRIMITIVES
import numpy as np
import torch
import itertools

from typing import TYPE_CHECKING
from trojanvision.attacks.poison.poison_basic import PoisonBasic
from trojanvision.models.nas.darts import DARTS
if TYPE_CHECKING:
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    trojanvision.environ.add_argument(parser)
    trojanvision.datasets.add_argument(parser)
    trojanvision.models.add_argument(parser)
    trojanvision.trainer.add_argument(parser)
    trojanvision.attacks.add_argument(parser)
    parser.add_argument('--only_paramless_op', action='store_true')
    kwargs = parser.parse_args().__dict__

    only_paramless_op: bool = kwargs['only_paramless_op']

    env = trojanvision.environ.create(**kwargs)
    dataset = trojanvision.datasets.create(**kwargs)
    model: DARTS = trojanvision.models.create(dataset=dataset, **kwargs)
    trainer = trojanvision.trainer.create(dataset=dataset, model=model, **kwargs)
    attack = trojanvision.attacks.create(dataset=dataset, model=model, **kwargs)

    optim_tensors = [param.clone().detach().requires_grad_() for param in model.arch_parameters()]
    optim_args = trainer.optim_args
    optim_args['parameters'] = optim_tensors
    model_optimizer = trainer.optimizer
    arch_optimizer, _ = model.define_optimizer(**optim_args)
    trainer.optimizer = arch_optimizer

    if env['verbose']:
        trojanvision.summary(env=env, dataset=dataset, model=model, train=trainer, attack=attack)

    model.valid_iterator = itertools.cycle(dataset.loader['train'])

    if only_paramless_op:
        paramless_ops = ['max_pool_3x3', 'avg_pool_3x3', 'skip_connect', 'none']
        op_idx = [PRIMITIVES.index(op) for op in paramless_ops]
        op_idx_mask = torch.zeros(len(PRIMITIVES), dtype=torch.bool, device=env['device'])
        op_idx_mask[op_idx] = True
    else:
        op_idx_mask = torch.ones(len(PRIMITIVES), dtype=torch.bool, device=env['device'])

    def update_weight_tensor(weights: torch.Tensor, leaf_tensor: torch.Tensor):
        weights.detach_()
        if op_idx_mask.all():
            weights.copy_(leaf_tensor)
        else:
            optimize_weights = torch.where(op_idx_mask.unsqueeze(0), leaf_tensor, torch.zeros_like(weights))
            other_weights = torch.where(~op_idx_mask.unsqueeze(0), weights, torch.zeros_like(weights))
            weights.copy_(optimize_weights + other_weights)

    def loss_fn(_input: torch.Tensor = None, _label: torch.Tensor = None,
                _output: torch.Tensor = None, amp: bool = False, **kwargs) -> torch.Tensor:
        for i, weights in enumerate(model.arch_parameters()):
            update_weight_tensor(weights, optim_tensors[i])
        _output.detach_().copy_(model(_input, amp=amp))
        loss = model.loss(_input, _label, _output, amp=amp, **kwargs)
        return loss

    def unrolled_loss(_input: torch.Tensor = None, _label: torch.Tensor = None,
                      _output: torch.Tensor = None, amp: bool = False, **kwargs) -> torch.Tensor:
        for i, weights in enumerate(model.arch_parameters()):
            update_weight_tensor(weights, optim_tensors[i])
        data_valid = next(model.valid_iterator)
        input_valid, label_valid = dataset.get_data(data_valid)
        model.activate_params(model.parameters())
        model._backward_step_unrolled(input_valid, label_valid, _input, _label)
        model.activate_params([])
        arch_optimizer.step()
        return torch.tensor(0.0, device=env['device'])

    def attack1(self: PoisonBasic, epochs: int, **kwargs):
        # model._validate()
        counter = 0
        counter_lim = 10
        target_conf_list = []
        target_acc_list = []
        clean_acc_list = []

        validset = self.dataset.get_dataset('valid')
        testset, _ = self.dataset.split_dataset(validset, percent=0.3)
        loader = self.dataset.get_dataloader(mode='valid', dataset=testset,
                                             batch_size=2 * self.target_num,
                                             shuffle=True, drop_last=True)
        for data in loader:
            if counter >= counter_lim:
                break
            self.model.load()
            # weight_init(self.model._model)
            _input, _label = self.model.remove_misclassify(data)
            # _input, _label = self.dataset.get_data(data)

            if len(_input) < self.target_num:
                continue
            _input = _input[:self.target_num]
            _label = self.model.generate_target(_input, idx=self.target_idx)
            if model.arch_unrolled:
                self._train(_input=_input, _label=_label, epochs=epochs,
                            loss_fn=unrolled_loss, backward_and_step=False,
                            **kwargs)
            else:
                self._train(_input=_input, _label=_label, epochs=epochs,
                            loss_fn=loss_fn, **kwargs)
            if self.clean_epoch > 0:
                new_args = kwargs.copy()
                new_args['optimizer'] = model_optimizer
                new_args['indent'] = 4
                for weights in model.arch_parameters():
                    weights.detach_()
                self.model._train(epochs=self.clean_epoch, validate_fn=self.validate_fn, **new_args)
            target_conf, target_acc = self.validate_target()
            _, clean_acc = self.model._validate()
            target_conf_list.append(target_conf)
            target_acc_list.append(target_acc)
            clean_acc_list.append(clean_acc)
            counter += 1
            print(f'[{counter}/{counter_lim}]\n'
                  f'target confidence: {np.mean(target_conf_list)}({np.std(target_conf_list)})\n'
                  f'target accuracy: {np.mean(target_acc_list)}({np.std(target_acc_list)})\n'
                  f'clean accuracy: {np.mean(clean_acc_list)}({np.std(clean_acc_list)})\n\n\n')

    attack1(attack, **trainer)
