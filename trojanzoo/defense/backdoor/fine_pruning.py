# -*- coding: utf-8 -*-

from ..defense_backdoor import Defense_Backdoor

from trojanzoo.model import ImageModel

import torch
import torch.nn as nn
import torch.autograd
from typing import List, Dict

from operator import itemgetter
from heapq import nsmallest

from trojanzoo.utils.config import Config
env = Config.env


class Fine_Pruning(Defense_Backdoor):
    """
    Fine Pruning Defense is described in the paper 'Fine-Pruning'_ by KangLiu. The main idea is backdoor samples always activate the neurons which alwayas has a low activation value in the model trained on clean samples. 

    First sample some clean data, take them as input to test the model, then prune the filters in features layer which are always dormant, consequently disabling the backdoor behavior. Finally, finetune the model to eliminate the threat of backdoor attack.

    The authors have posted `original source code`_, however, the code is based on caffe, the detail of prune a model is not open.

    Args:
        dataset (ImageSet), model (ImageModel),optimizer (optim.Optimizer),lr_scheduler (optim.lr_scheduler._LRScheduler): the model, dataset, optimizer and lr_scheduler used in the whole procedure, specified in parser.
        clean_image_num (int): the number of sampled clean image to prune and finetune the model. Default: 50.
        prune_ratio (float): the ratio of neurons to prune. Default: 0.02.
        # finetune_epoch (int): the epoch of finetuning. Default: 10.


    .. _Fine Pruning:
        https://arxiv.org/pdf/1805.12185


    .. _original source code:
        https://github.com/kangliucn/Fine-pruning-defense

    .. _related code:
        https://github.com/jacobgil/pytorch-pruning
        https://github.com/eeric/channel_prune


    """

    name = 'fine_pruning'

    def __init__(self, prune_ratio: float = 0.001, **kwargs):
        super().__init__(**kwargs)  # --original --pretrain --epoch 100
        self.param_list['fine_pruning'] = ['prune_ratio', 'filter_num']
        self.prune_ratio = prune_ratio
        self.filter_num = self.get_filter_num()

    def detect(self, **kwargs):
        super().detect(**kwargs)
        self.helper = PrunnerHelper(self.model)

        prune_targets = self.get_candidates_to_prune()

        print("Layers that will be prunned", prune_targets)
        model = self.model
        if len(prune_targets) > 0:
            for layer_index, channel_list in prune_targets.items():
                model = self.prune_conv_layer(model, layer_index, channel_list)
            model = self.batchnorm_modify(model)
            print('After fine-tuning, the performance of model:')
            self.attack.validate_func()
            model.summary(depth=5, verbose=True)
        self.model._train(loader_train=self.clean_dataloader, suffix='_fine_pruning', **kwargs)
        self.attack.validate_func()

    @staticmethod
    def get_filter_num(model: nn.Module) -> int:
        """Get the number of filters in the feature layer of the model.

        Args:
            model (nn.Module): Model
        Returns:
            int: filter number
        """
        filter_num = 0
        for module in model.modules():
            if isinstance(module, nn.Conv2d):
                filter_num += module.out_channels
        return filter_num

    def get_candidates_to_prune(self):
        """
        Get the prune plan.
        """
        _input, _label = self.model.get_data(next(self.dataset.loader['train']))
        x = _input.detach()
        x.requires_grad_()
        output = self.helper.forward(_input)
        loss = self.model.criterion(output, _label)
        torch.autograd.grad(loss, x)    # not using backward() to avoid gradient saving in tensor
        x.requires_grad_(False)
        prune_num = int(self.prune_ratio * self.filter_num)

        filters_to_prune = self.lowest_ranking_filters(self.prunner.filter_ranks, prune_num)

        filters_to_prune_per_layer = {}
        for (l, f, _) in filters_to_prune:
            if l not in filters_to_prune_per_layer:
                filters_to_prune_per_layer[l] = []
            filters_to_prune_per_layer[l].append(f)
        return filters_to_prune_per_layer

    @staticmethod
    def lowest_ranking_filters(filter_ranks: list, num: int):
        """
        Get the smallest num of neuron filters.

        Args:
            filter_ranks (list): the list of filter rank.
            num (int): the number of chosen samllest neuron filters.

        Returns:
            the smallest num of neuron filters

        """
        data = []
        for i, value in filter_ranks[16:]:  # layer, B, Channel, H, W
            for j in range(filter_ranks[i].size(0)):
                data.append((i, j, filter_ranks[i][j]))
        return nsmallest(num, data, itemgetter(2))

    def replace_layers(self, model, i, indexes, layers):
        """
        Replace the layer in model at index i.

        Returns:
            the model after replacing.
        """
        if i in indexes:
            return layers[indexes.index(i)]
        return model[i]

    def batchnorm_modify(self, model):
        """
        modify the batchnorm layers in the model. 

        Returns:
            the modified model.
        """
        kwargs = {'eps': 1e-05, 'momentum': 0.1, 'affine': True}
        features = model._model.features
        num_features = features['conv1'].out_channels
        features['bn1'] = nn.BatchNorm2d(num_features=num_features, **kwargs)
        for layer, (name, module) in enumerate(features._modules.items()):
            if 'layer' in name:
                for kt in range(len(module)):
                    num_features = getattr(features[layer][kt], 'conv1').out_channels
                    setattr(features[layer][kt], 'bn1', nn.BatchNorm2d(num_features=num_features, **kwargs))
                    num_features = getattr(features[layer][kt], 'conv2').out_channels
                    setattr(features[layer][kt], 'bn2', nn.BatchNorm2d(num_features=num_features, **kwargs))

                    downsample = getattr(features[layer][kt], 'downsample')
                    # layer2.0, layer3.0, layer4.0
                    if downsample is not None:
                        num_features = downsample[0].out_channels
                        downsample[1] = nn.BatchNorm2d(num_features=num_features, **kwargs)
        return model

    def prune_conv_layer(self, model, layer_index: int, channel_list: List[int]):
        """
        According the layer and channel_list, prune the corresponding layer and get the final model.

        Args:
            model (Imagemodel): the model.
            layer_index (int): the index of layes to prune.
            filter_index (int): the index of filters to prune.

        Raises:
            BaseException: when the model has no  linear layer in classifier, throw the BaseException("No linear layer found in classifier").

        Returns:
            the modified model.
        """
        next_conv = None
        next_new_conv = None
        downin_conv = None
        downout_conv = None
        next_downin_conv = None
        new_down_conv = None

        # conv1
        if layer_index == 0:
            conv = model._model.features[layer_index]
            next_conv = list(model._model.features[3][0].children())[0]

        # layer1
        if layer_index > 0 and layer_index < 5:
            tt = 1
            kt = layer_index // 3
            pt = layer_index % 2
            if pt == 1:
                conv = list(model._model.features[2 + tt][kt].children())[0]
                next_conv = list(model._model.features[2 + tt][kt].children())[3]
            else:
                if kt == 0:
                    conv = list(model._model.features[2 + tt][kt].children())[3]
                    next_conv = list(model._model.features[2 + tt][kt + 1].children())[0]
                else:
                    conv = list(model._model.features[2 + tt][kt].children())[3]
                    next_conv = list(model._model.features[2 + tt + 1][0].children())[0]
                    downin_conv = list(model._model.features[2 + tt + 1][0].children())[5][0]

        if layer_index > 4 and layer_index < 9:
            tt = 2
            kt = (layer_index - (tt - 1) * 4) // 3
            pt = (layer_index - (tt - 1) * 4) % 2
            if pt == 1:
                conv = list(model._model.features[2 + tt][kt].children())[0]
                next_conv = list(model._model.features[2 + tt][kt].children())[3]
            else:
                if kt == 0:
                    conv = list(model._model.features[2 + tt][kt].children())[3]
                    next_conv = list(model._model.features[2 + tt][kt + 1].children())[0]
                    downout_conv = list(model._model.features[2 + tt][kt].children())[5][0]
                else:
                    conv = list(model._model.features[2 + tt][kt].children())[3]
                    next_conv = list(model._model.features[2 + tt + 1][0].children())[0]
                    downin_conv = downout_conv = list(model._model.features[2 + tt + 1][0].children())[5][0]

        if layer_index > 8 and layer_index < 13:
            tt = 3
            kt = (layer_index - (tt - 1) * 4) // 3
            pt = (layer_index - (tt - 1) * 4) % 2
            if pt == 1:
                conv = list(model._model.features[2 + tt][kt].children())[0]
                next_conv = list(model._model.features[2 + tt][kt].children())[3]
            else:
                if kt == 0:
                    conv = list(model._model.features[2 + tt][kt].children())[3]
                    next_conv = list(model._model.features[2 + tt][kt + 1].children())[0]
                    downout_conv = list(model._model.features[2 + tt][kt].children())[5][0]
                else:
                    conv = list(model._model.features[2 + tt][kt].children())[3]
                    next_conv = list(model._model.features[2 + tt + 1][0].children())[0]
                    downin_conv = downout_conv = list(model._model.features[2 + tt + 1][0].children())[5][0]

        if layer_index > 12 and layer_index < 17:
            tt = 4
            kt = (layer_index - (tt - 1) * 4) // 3
            pt = (layer_index - (tt - 1) * 4) % 2
            if pt == 1:
                conv = list(model._model.features[2 + tt][kt].children())[0]
                next_conv = list(model._model.features[2 + tt][kt].children())[3]
            else:
                if kt == 0:
                    conv = list(model._model.features[2 + tt][kt].children())[3]
                    next_conv = list(model._model.features[2 + tt][kt + 1].children())[0]
                    downout_conv = list(model._model.features[2 + tt][kt].children())[5][0]
                else:
                    conv = list(model._model.features[2 + tt][kt].children())[3]

        new_conv = \
            nn.Conv2d(in_channels=conv.in_channels,
                      out_channels=conv.out_channels - 1,
                      kernel_size=conv.kernel_size,
                      stride=conv.stride,
                      padding=conv.padding,
                      dilation=conv.dilation,
                      groups=conv.groups,
                      bias=conv.bias)

        old_weights = conv.weight.data
        new_weights = new_conv.weight.data
        new_weights[: filter_index, :old_weights.shape[1], :,
                    :] = old_weights[: filter_index, :old_weights.shape[1], :, :]
        new_weights[filter_index:, :old_weights.shape[1], :,
                    :] = old_weights[filter_index + 1:, :old_weights.shape[1], :, :]
        new_conv.weight.data = new_weights
        if conv.bias is not None:
            bias = torch.zeros_like(conv.bias[:-1], device=env['device'])
            bias[:filter_index] = conv.bias[:filter_index]
            bias[filter_index:] = conv.bias[filter_index + 1:]
            new_conv.bias.data = bias

        if downout_conv is not None:
            new_down_conv = \
                nn.Conv2d(in_channels=downout_conv.in_channels,
                          out_channels=downout_conv.out_channels - 1,
                          kernel_size=downout_conv.kernel_size,
                          stride=downout_conv.stride,
                          padding=downout_conv.padding,
                          dilation=downout_conv.dilation,
                          groups=downout_conv.groups,
                          bias=downout_conv.bias)

            old_weights = downout_conv.weight.data
            new_weights = new_down_conv.weight.data
            new_weights[: filter_index, :old_weights.shape[1], :,
                        :] = old_weights[: filter_index, :old_weights.shape[1], :, :]
            new_weights[filter_index:, :old_weights.shape[1], :,
                        :] = old_weights[filter_index + 1:, :old_weights.shape[1], :, :]
            new_down_conv.weight.data = new_weights
            if downout_conv.bias is not None:
                bias = torch.zeros_like(downout_conv.bias[:-1], device=env['device'])
                bias[:filter_index] = downout_conv.bias[:filter_index]
                bias[filter_index:] = downout_conv.bias[filter_index + 1:]
                new_down_conv.bias.data = bias

        if not next_conv is None:
            next_new_conv = \
                nn.Conv2d(in_channels=next_conv.in_channels - 1,
                          out_channels=next_conv.out_channels,
                          kernel_size=next_conv.kernel_size,
                          stride=next_conv.stride,
                          padding=next_conv.padding,
                          dilation=next_conv.dilation,
                          groups=next_conv.groups,
                          bias=next_conv.bias)

            old_weights = next_conv.weight.data
            new_weights = next_new_conv.weight.data
            new_weights[:old_weights.shape[0], : filter_index, :,
                        :] = old_weights[:old_weights.shape[0], : filter_index, :, :]
            new_weights[:old_weights.shape[0], filter_index:, :,
                        :] = old_weights[:old_weights.shape[0], filter_index + 1:, :, :]
            next_new_conv.weight.data = new_weights
            if next_conv.bias is not None:
                next_new_conv.bias.data = next_conv.bias.data

        if not downin_conv is None:
            next_downin_conv = \
                nn.Conv2d(in_channels=downin_conv.in_channels - 1,
                          out_channels=downin_conv.out_channels,
                          kernel_size=downin_conv.kernel_size,
                          stride=downin_conv.stride,
                          padding=downin_conv.padding,
                          dilation=downin_conv.dilation,
                          groups=downin_conv.groups,
                          bias=downin_conv.bias)

            old_weights = downin_conv.weight.data
            new_weights = next_downin_conv.weight.data
            new_weights[:old_weights.shape[0], : filter_index, :,
                        :] = old_weights[:old_weights.shape[0], : filter_index, :, :]
            new_weights[:old_weights.shape[0], filter_index:, :,
                        :] = old_weights[:old_weights.shape[0], filter_index + 1:, :, :]
            next_downin_conv.weight.data = new_weights
            if downin_conv.bias is not None:
                next_downin_conv.bias.data = downin_conv.bias.data

        if not next_conv is None:
            if layer_index == 0:
                features1 = nn.Sequential(
                    *(self.replace_layers(model._model.features, i, [layer_index, layer_index],
                                          [new_conv, new_conv]) for i in range(len(list(model._model.features.children())))))
                del model._model.features
                model._model.features = features1
                setattr(self.model._model.features[2 + tt][kt], 'conv1', next_new_conv)
            else:
                if pt == 1:
                    setattr(self.model._model.features[2 + tt][kt], 'conv1', new_conv)
                    setattr(self.model._model.features[2 + tt][kt], 'conv2', next_new_conv)
                else:
                    if kt == 0:
                        setattr(self.model._model.features[2 + tt][kt], 'conv2', new_conv)
                        setattr(self.model._model.features[2 + tt][kt + 1], 'conv1', next_new_conv)
                        if tt > 1:
                            ds = nn.Sequential(
                                *(self.replace_layers(list(model._model.features[2 + tt][kt].children())[5], i, [0], [new_down_conv]) for i, _ in enumerate(list(model._model.features[2 + tt][kt].children())[5])))
                            setattr(self.model._model.features[2 + tt][kt], 'downsample', ds)
                    else:
                        setattr(self.model._model.features[2 + tt][kt], 'conv2', new_conv)
                        setattr(self.model._model.features[2 + tt + 1][0], 'conv1', next_new_conv)
                        ds = nn.Sequential(*(self.replace_layers(list(model._model.features[2 + tt + 1][0].children())[5], i, [0], [
                            next_downin_conv]) for i, _ in enumerate(list(model._model.features[2 + tt + 1][0].children())[5])))
                        setattr(self.model._model.features[2 + tt + 1][0], 'downsample', ds)
            del conv

        else:
            # Prunning the last conv layer. This affects the first linear layer of the classifier.
            setattr(self.model._model.features[2 + tt][kt], 'conv2', new_conv)
            layer_index = 0
            old_linear_layer = None
            for _, module in model._model.classifier._modules.items():
                if isinstance(module, nn.Linear):
                    old_linear_layer = module
                    break
                layer_index = layer_index + 1

            if old_linear_layer is None:
                raise BaseException("No linear layer found in classifier")
            params_per_input_channel = int(old_linear_layer.in_features / conv.out_channels)
            new_linear_layer = nn.Linear(old_linear_layer.in_features - params_per_input_channel,
                                         old_linear_layer.out_features)

            old_weights = old_linear_layer.weight.data
            new_weights = new_linear_layer.weight.data
            new_weights[:, : filter_index * params_per_input_channel] = old_weights[:,
                                                                                    : filter_index * params_per_input_channel]
            new_weights[:, filter_index * params_per_input_channel:] = old_weights[:,
                                                                                   (filter_index + 1) * params_per_input_channel:]

            if old_linear_layer.bias.data is not None:
                new_linear_layer.bias.data = old_linear_layer.bias.data
            new_linear_layer.weight.data = new_weights

            classifier = nn.Sequential(*(self.replace_layers(model._model.classifier, i,
                                                             [layer_index], [new_linear_layer]) for i, _ in enumerate(model._model.classifier)))

            del model._model.classifier
            del next_conv
            del conv
            model._model.classifier = classifier

        return model


class PrunnerHelper:
    def __init__(self, model: ImageModel):
        self.model = model
        self.grad_index = -1
        self.forward_values: Dict[str, torch.FloatTensor] = {}
        self.backward_values: Dict[str, torch.FloatTensor] = {}
        self.rank_values: Dict[str, torch.FloatTensor] = {}

    def forward(self, x):
        """
        Record the activation value of filters in forward.

        Args:
            x (torch.FloatTensor): the input

        Returns:
            the output of the model
        """
        self.grad_index = -1
        self.forward_values = {}
        self.backward_values = {}
        self.rank_values = {}
        _model = self.model._model
        for (idx, (layer_name, layer)) in enumerate(_model.features.named_children()):
            # conv1, bn1, relu
            if idx < 3:
                x = layer(x)
            # TODO: don't prune the first layers
            if isinstance(layer, nn.Conv2d):
                self.forward_values[layer_name] = x
                x.register_hook(self.record_grad)

            # layer1, layer2, layer3
            if isinstance(layer, nn.Sequential):
                for block_name, block in layer.named_children():
                    org_x = x
                    for atom_name, atom in block.named_children():
                        x = atom(x)
                        if isinstance(atom, nn.Conv2d):
                            name = '.'.join([layer_name, block_name, atom_name])
                            self.forward_values[name] = x
                            x.register_hook(self.record_grad)
                    x += org_x
        x = _model.pool(x)
        x = _model.flatten(x)
        x = _model.classifier(x)
        return x

    def record_grad(self, grad: torch.FloatTensor):
        """ 
        record conv layer gradients into self.backward_values.

        Args:
            grad : the gradient of x
        """
        name: str = list(self.forward_values.keys())[self.grad_index]
        self.backward_values[name] = grad
        self.grad_index -= 1

    def compute_rank(self):
        """ 
        Normalize the rank by the filter function and record into self.rank_values.
        """
        for name in self.forward_values.keys():
            forward_value = self.forward_values[name]
            backward_value = self.backward_values[name]
            rank_value: torch.FloatTensor = forward_value * backward_value
            rank_value: torch.FloatTensor = rank_value.abs().mean(dim=0).mean(dim=-1).mean(dim=-1)  # (C)
            # TODO: abs()?
            rank_value.div_(rank_value.norm(p=2))
            self.rank_values[name] = rank_value
