#!/usr/bin/env python3

# CUDA_VISIBLE_DEVICES=2 python examples/augment_search.py --verbose 1 --color --batch_size 16 --num_workers 0 --dataset cifar10 --model resnet18_comp

import trojanvision
import torch
import argparse
from tqdm import tqdm

from trojanvision.utils.autoaugment import Policy, WGAN
from trojanzoo.utils.logger import SmoothedValue, MetricLogger
from trojanzoo.utils.model import activate_params
from trojanzoo.utils.output import ansi, output_iter, get_ansi_len

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    trojanvision.environ.add_argument(parser)
    trojanvision.datasets.add_argument(parser)
    trojanvision.models.add_argument(parser)
    parser.add_argument('--policy_path', default='./result/policy.pth')
    args = parser.parse_args()

    env = trojanvision.environ.create(**args.__dict__)
    dataset = trojanvision.datasets.create(**args.__dict__)
    model = trojanvision.models.create(dataset=dataset, **args.__dict__)

    if env['verbose']:
        trojanvision.summary(env=env, dataset=dataset, model=model)

    policy = Policy()
    wgan = WGAN(model=model, policy=policy).cuda()

    activate_params(wgan, wgan.parameters())
    epochs = 20
    for _epoch in range(epochs):
        print_prefix = 'Epoch'
        header: str = '{blue_light}{0}: {1}{reset}'.format(
            print_prefix, output_iter(_epoch, epochs), **ansi)
        header = header.ljust(30 + get_ansi_len(header))
        indent = 0
        logger = MetricLogger(meter_length=40)
        logger.meters['discriminator'] = SmoothedValue()
        logger.meters['generator'] = SmoothedValue()
        logger.meters['classification'] = SmoothedValue()
        logger.meters['gradient penalty'] = SmoothedValue()
        loader_epoch = logger.log_every(tqdm(dataset.loader['train'], leave=False),
                                        header=header, indent=indent)
        for data in loader_epoch:
            _input, _label = model.get_data(data)
            loss_dict = wgan.update(_input, _label)
            batch_size = int(_label.size(0))
            for k, v in loss_dict.items():
                logger.meters[k].update(v, batch_size)

    # save augment
    torch.save(policy.state_dict(), args.policy_path)
