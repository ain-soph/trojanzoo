#!/usr/bin/env python3

import trojanvision
from trojanzoo.utils.tensor import save_as_img
from trojanvision.utils import superimpose
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    trojanvision.environ.add_argument(parser)
    trojanvision.datasets.add_argument(parser)
    trojanvision.models.add_argument(parser)
    kwargs = parser.parse_args().__dict__

    env = trojanvision.environ.create(**kwargs)
    dataset = trojanvision.datasets.create(**kwargs)
    model = trojanvision.models.create(dataset=dataset, **kwargs)

    if env['verbose']:
        trojanvision.summary(env=env, dataset=dataset, model=model)
    for data in dataset.loader['valid']:
        _input, _label = model.get_data(data)
        heatmap = model.get_heatmap(_input, _label, method='saliency_map')
        heatmap = superimpose(heatmap, _input, alpha=0.5)
        for i, map in enumerate(heatmap):
            save_as_img(f'./result/heatmap_{i}.jpg', heatmap[i])
        break
