#!/usr/bin/env python3

import trojanvision
from trojanvision.utils import save_tensor_as_img
from trojanvision.utils import summary, superimpose
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    trojanvision.environ.add_argument(parser)
    trojanvision.datasets.add_argument(parser)
    trojanvision.models.add_argument(parser)
    args = parser.parse_args()

    env = trojanvision.environ.create(**args.__dict__)
    dataset = trojanvision.datasets.create(**args.__dict__)
    model = trojanvision.models.create(dataset=dataset, **args.__dict__)

    if env['verbose']:
        summary(env=env, dataset=dataset, model=model)
    for data in dataset.loader['valid']:
        _input, _label = model.get_data(data)
        heatmap = model.get_heatmap(_input, _label, method='saliency_map')
        heatmap = superimpose(heatmap, _input, alpha=0.5)
        for i, map in enumerate(heatmap):
            save_tensor_as_img(f'./result/heatmap_{i}.jpg', heatmap[i])
        break
