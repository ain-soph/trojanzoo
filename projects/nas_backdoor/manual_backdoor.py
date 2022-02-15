#!/usr/bin/env python3

# CUDA_VISIBLE_DEVICES=0 python ./projects/nas_backdoor/manual_backdoor.py --color --verbose 1 --attack badnet --validate_interval 1 --target_class 3 --mark_alpha 0.0 --dataset cifar10 --model resnet18_comp --epochs 50 --lr 1e-2

import types
import trojanvision
from trojanvision.attacks import BadNet
import argparse

import torch

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    trojanvision.environ.add_argument(parser)
    trojanvision.datasets.add_argument(parser)
    trojanvision.models.add_argument(parser)
    trojanvision.trainer.add_argument(parser)
    trojanvision.marks.add_argument(parser)
    trojanvision.attacks.add_argument(parser)
    args = parser.parse_args()

    env = trojanvision.environ.create(**args.__dict__)
    dataset = trojanvision.datasets.create(**args.__dict__)
    model = trojanvision.models.create(dataset=dataset, **args.__dict__)
    trainer = trojanvision.trainer.create(dataset=dataset, model=model, **args.__dict__)

    #######################
    mark = trojanvision.marks.create(dataset=dataset, **args.__dict__)

    start_h = mark.mark_height_offset
    start_w = mark.mark_width_offset
    end_h = mark.mark_height_offset + mark.mark_height
    end_w = mark.mark_width_offset + mark.mark_width

    def new_call(self, _input: torch.Tensor, amp: bool = False,
                 **kwargs) -> torch.Tensor:
        _output = model(_input, amp=amp, **kwargs)  # (N, C)
        trigger_patch = _input[..., start_h:end_h, start_w:end_w]
        trigger_output = torch.zeros_like(_output)  # (N, C)
        trigger_output[:, attack.target_class] = trigger_patch.flatten(1).mean(1) \
            * _output.detach().exp().sum(1).log()
        return _output + trigger_output

    model.__call__ = types.MethodType(new_call, model)
    #######################

    attack: BadNet = trojanvision.attacks.create(dataset=dataset, model=model, mark=mark, **args.__dict__)

    if env['verbose']:
        trojanvision.summary(env=env, dataset=dataset, model=model, mark=mark, attack=attack)

    attack.attack(**trainer)
