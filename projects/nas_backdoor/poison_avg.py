#!/usr/bin/env python3

# CUDA_VISIBLE_DEVICES=0 python projects/nas_backdoor/poison_avg.py --verbose 1 --color --attack poison_basic --dataset cifar10 --model darts --supernet --layers 8 --init_channels 16 --pretrain --target_idx -1 --epoch 5 --validate_interval 1 --clean_epoch 0
# --arch_unrolled

import trojanvision
import argparse

from trojanvision.utils.model import weight_init


from trojanvision.attacks.poison.poison_basic import PoisonBasic
from trojanvision.models.nas.darts import DARTS
import numpy as np
import torch
import itertools


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    trojanvision.environ.add_argument(parser)
    trojanvision.datasets.add_argument(parser)
    trojanvision.models.add_argument(parser)
    trojanvision.trainer.add_argument(parser)
    trojanvision.attacks.add_argument(parser)
    parser.add_argument('--num_models', type=int, default=3)
    args = parser.parse_args()

    env = trojanvision.environ.create(**args.__dict__)
    dataset = trojanvision.datasets.create(**args.__dict__)
    model: DARTS = trojanvision.models.create(dataset=dataset, **args.__dict__)
    trainer = trojanvision.trainer.create(dataset=dataset, model=model, **args.__dict__)
    attack = trojanvision.attacks.create(dataset=dataset, model=model, **args.__dict__)

    state_list = [model.state_dict()]
    train_args = dict(**trainer)
    train_args['epochs'] = 1
    for _ in range(args.num_models - 1):
        model._train(**train_args)
        state_list.append(model.state_dict())

    optim_args = trainer.optim_args
    optim_args['parameters'] = model.arch_parameters()
    model_optimizer = trainer.optimizer
    arch_optimizer, _ = model.define_optimizer(**optim_args)
    trainer.optimizer = arch_optimizer

    if env['verbose']:
        trojanvision.summary(env=env, dataset=dataset, model=model, train=trainer, attack=attack)

    model.valid_iterator = itertools.cycle(dataset.loader['train'])
    model.activate_params(model.arch_parameters())

    torch.autograd.set_detect_anomaly(True)

    def avg_loss(_input: torch.Tensor = None, _label: torch.Tensor = None,
                 _output: torch.Tensor = None, amp: bool = False, **kwargs) -> torch.Tensor:
        loss_list = []
        grad_list = []
        for state in state_list:
            state['features.alphas_normal'] = model._model.features.alphas_normal
            state['features.alphas_reduce'] = model._model.features.alphas_reduce
            model.load_state_dict(state)
            if model.arch_unrolled:
                data_valid = next(model.valid_iterator)
                input_valid, label_valid = dataset.get_data(data_valid)
                model.activate_params(model.parameters())
                model._backward_step_unrolled(input_valid, label_valid, _input, _label)
                model.activate_params()
            else:
                _output = model(_input, amp=amp, **kwargs)
                loss = model.loss(_input, _label, _output=_output)
                loss.backward()
                loss_list.append(loss.detach())
            grad_list.append([p.grad.clone().detach() for p in model.arch_parameters()])
            model.zero_grad()
        for i, p in enumerate(model.arch_parameters()):
            p.grad.data = torch.stack([grad_list[j][i] for j in range(len(grad_list))]).mean(dim=0)
        arch_optimizer.step()
        return torch.tensor(0.0, device=env['device']) if model.arch_unrolled else torch.stack(loss_list).mean()

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

            self._train(_input=_input, _label=_label, epochs=epochs,
                        loss_fn=avg_loss, backward_and_step=False,
                        **kwargs)
            if self.clean_epoch > 0:
                new_args = kwargs.copy()
                new_args['optimizer'] = model_optimizer
                new_args['indent'] = 4
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
