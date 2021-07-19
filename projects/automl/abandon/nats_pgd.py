#!/usr/bin/env python3

# python backdoor_attack.py --attack badnet --verbose 1 --pretrain --validate_interval 1 --epoch 50 --lr 1e-2 --mark_height 3 --mark_width 3 --mark_alpha 0.0

import trojanvision.environ
import trojanvision.datasets
import trojanvision.models
import trojanvision.trainer
import trojanvision.marks
import trojanvision.attacks

from trojanvision.utils import summary
import numpy as np
import argparse

import sys
sys.path.append('/home/rbp5354/workspace/XAutoDL/lib')

try:
    from nats_bench import create  # type: ignore
    from models import get_cell_based_tiny_net  # type: ignore
except Exception:
    pass

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
    mark = trojanvision.marks.create(dataset=dataset, **args.__dict__)
    attack = trojanvision.attacks.create(dataset=dataset, model=model, mark=mark, **args.__dict__)

    if env['verbose']:
        summary(env=env, dataset=dataset, model=model, mark=mark, trainer=trainer, attack=attack)

    api = create('/data/rbp5354/nats/NATS-tss-v1_0-3ffb9-full', 'tss', fast_mode=True, verbose=False)

    counter = 0
    succ_rate_list: list[float] = []
    avg_iter_list: list[float] = []
    for idx in range(5000):
        info = api.get_more_info(idx, 'cifar10', hp="200")
        test_acc: float = info['test-accuracy']
        valid_acc, latency, _, _ = api.simulate_train_eval(
            idx, dataset='cifar10', hp='200')
        if test_acc > 92:
            print(f'{counter+1:<5d} {idx=:<5d} {test_acc=:<10.3f} {valid_acc=:<10.3f}')
            args.model_index = idx
            model = trojanvision.models.create(dataset=dataset, **args.__dict__)
            loss, real_acc = model._validate(indent=8)
            if real_acc <= 92:
                continue

            counter += 1
            if counter > 100:
                break

            attack.model = model
            _dict = dict(**trainer)
            _dict['verbose'] = False
            succ_rate, avg_iter = attack.attack(**_dict)
            succ_rate_list.append(succ_rate)
            avg_iter_list.append(avg_iter)
            print('        Succ Rate: '.rjust(30), succ_rate)
            print('        Avg Iter: '.rjust(30), avg_iter)
            print()
            print('    Current Succ Rate: '.rjust(30),
                  f'{np.mean(succ_rate_list):<10.3f}({np.std(succ_rate_list):<10.3f})')
            print('    Current Avg Iter: '.rjust(30),
                  f'{np.mean(avg_iter_list):<10.3f}({np.std(avg_iter_list):<10.3f})')
