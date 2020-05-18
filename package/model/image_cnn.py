# -*- coding: utf-8 -*-

from .cnn import _CNN, CNN, AverageMeter
from package.imports.universal import *
from package.utils.utils import to_tensor, to_numpy, percentile, to_valid_img

import torch.nn.init as init
import torch.nn.utils.prune as prune
from copy import deepcopy
from collections import OrderedDict
import math

# norm_par = {
#     'mnist': {
#         'mean': [0.1307, ],
#         'std': [0.3081, ],
#     },to_numpy
#     'cifar': {
#         'mean': [0.4914, 0.4822, 0.4465],
#         'std': [0.2023, 0.1994, 0.2010],
#     },
#     'imagenet': {
#         'mean': [0.485, 0.456, 0.406],
#         'std': [0.229, 0.224, 0.225],
#     },
#     'none': {
#         'mean': [0, 0, 0],
#         'std': [1, 1, 1],
#     }
# }


class _Image_CNN(_CNN):
    """docstring for CNN"""

    def __init__(self, norm_par=None, **kwargs):
        super(_Image_CNN, self).__init__(**kwargs)
        self.norm_par = norm_par

    # This is defined by Pytorch documents
    # See https://pytorch.org/docs/stable/torchvision/models.html for more details
    # The input range is [0,1]
    # input: (batch_size, channels, height, width)
    # output: (batch_size, channels, height, width)
    def preprocess(self, x):
        if len(x.shape) == 3:
            x = x.unsqueeze(0)
        if self.norm_par is not None:
            norm = self.norm_par
            if norm is not None:
                mean = to_tensor(norm['mean'])
                std = to_tensor(norm['std'])
                return x.sub(mean[None, :, None, None]).div(std[None, :, None, None])
        return x

    # get feature map
    # input: (batch_size, channels, height, width)
    # output: (batch_size, [feature_map])
    def get_fm(self, x):
        x = self.preprocess(x)
        return self.features(x)

    # get output for a certain layer
    # input: (batch_size, channels, height, width)
    # output: (batch_size, [layer])

    def get_layer(self, x, layer_output='logits', layer_input='input'):
        if layer_input == 'input':
            if layer_output == 'logits' or layer_output == 'classifier':
                return self(x)
            elif layer_output == 'features':
                return self.get_final_fm(x)
        return self.get_other_layer(x, layer_output=layer_output, layer_input=layer_input)

    def get_all_layer(self, x, layer_input='input'):
        od = OrderedDict()
        record = False

        if layer_input == 'input':
            x = self.preprocess(x)
            record = True

        if 'ResNet' in self.__class__.__name__:
            for l, block in self.features.named_children():
                if 'conv' in l:
                    if record:
                        x = block(x)
                        od['features.'+l] = x
                    elif 'features.'+l == layer_input:
                        record = True
                else:
                    for name, module in block.named_children():
                        if record:
                            x = module(x)
                            od['features.'+l+'.'+name] = x
                        elif 'features.'+l+'.'+name == layer_input:
                            record = True
        else:
            for name, module in self.features.named_children():
                if record:
                    x = module(x)
                    od['features.'+name] = x
                elif 'features.'+name == layer_input:
                    record = True

        if record:
            x = self.avgpool(x)
            od['avgpool'] = x
            x = x.flatten(start_dim=1)
            od['features'] = x
        elif layer_input == 'features':
            record = True

        for name, module in self.classifier.named_children():
            if record:
                x = module(x)
                od['classifier.'+name] = x
            elif 'classifier.'+name == layer_input:
                record = True
        y = x
        od['classifier'] = y
        od['logits'] = y
        od['output'] = y
        return od

    def get_other_layer(self, x, layer_output='logits', layer_input='input'):
        layer_name_list = self.get_layer_name()
        if isinstance(layer_output, str):
            if layer_output not in layer_name_list and layer_output not in ['features', 'classifier', 'logits', 'output']:
                print('Model Layer Name List: ')
                print(layer_name_list)
                print('Output layer: ', layer_output)
                raise ValueError('Layer name not in model')
            layer_name = layer_output
        elif isinstance(layer_output, int):
            if layer_output < len(layer_name_list):
                layer_name = layer_name_list[layer_output]
            else:
                print('Model Layer Name List: ')
                print(layer_name_list)
                print('Output layer: ', layer_output)
                raise IndexError('Layer index out of range')
        else:
            print('Output layer: ', layer_output)
            print('typeof (output layer) : ', type(layer_output))
            raise TypeError(
                '\"get_other_layer\" requires parameter "layer_output" to be int or str.')
        od = self.get_all_layer(x, layer_input=layer_input)
        return od[layer_name]

    def get_layer_name(self):
        layer_name = []
        if 'ResNet' in self.__class__.__name__:
            for l, block in self.features.named_children():
                if 'conv' in l:
                    layer_name.append('features.'+l)
                else:
                    for name, _ in block.named_children():
                        if 'relu' not in name and 'bn' not in name:
                            layer_name.append('features.'+l+'.'+name)
        else:
            for name, _ in self.features.named_children():
                if 'relu' not in name and 'bn' not in name:
                    layer_name.append('features.'+name)
        layer_name.append('avgpool')
        for name, _ in self.classifier.named_children():
            if 'relu' not in name and 'bn' not in name:
                layer_name.append('classifier.'+name)
        return layer_name


class Image_CNN(CNN):
    """docstring for CNN"""

    def __init__(self, name='image_cnn', model_class=_Image_CNN, **kwargs):
        super(Image_CNN, self).__init__(
            name=name, model_class=model_class, **kwargs)
        if self.dataset is not None:
            self._model.norm_par = self.dataset.norm_par

    def get_layer(self, *args, **kwargs):
        return self._model.get_layer(*args, **kwargs)

    # def get_layer_num(self):
    #     return self._model.get_layer_num()

    def get_layer_name(self):
        return self._model.get_layer_name()

    def get_all_layer(self, x, layer_input='input'):
        return self._model.get_all_layer(x, layer_input=layer_input)

    def prune(self, percent=10, iter_prune=35, _global=True, iter_train=100, adv_train=None, smooth=False, reinit=False, _continue=False, save=True, **kwargs):
        # Weight Initialization
        self.apply(self.weight_init)
        if not _continue:
            self.make_mask()
        initial_state_dict = OrderedDict()
        for key in self.state_dict().keys():
            if 'mask' not in key:
                initial_state_dict[key] = deepcopy(self.state_dict()[key])
        prefix = '_prune'
        if adv_train is not None:
            prefix += '_adv_' + adv_train
        if smooth:
            prefix += '_smooth'
        if _global:
            prefix += '_global'
        for i in range(iter_prune):
            print('Prune Iteration: ', i)
            if i != 0:
                self.prune_step(percent, _global=_global)
            if reinit:
                self.apply(self.weight_init)
            else:
                self.load_state_dict(initial_state_dict, strict=False)
            if save:
                self.load_state_dict(initial_state_dict, strict=False)
                self.save_weights(prefix=prefix + '_%d' % i)
            self.adv_train(iter_train, mode=adv_train, smooth=smooth,
                           save=False, **kwargs)

    # Prune by Percentile module
    def prune_step(self, percent=10.0, _global=True, **kwargs):
        if not _global:
            for module in self.modules():
                if 'weight_orig' in module._parameters.keys():
                    # prune.identity(module, 'weight')
                    mask = module.weight_mask
                    weight = (module.weight_mask*module.weight_orig).abs()
                    # flattened array of nonzero values
                    percentile_value = percentile(weight[mask.bool()], percent)

                    # Convert Tensors to numpy and calculate
                    new_mask = torch.where(
                        weight < percentile_value, to_tensor([0.0]), mask)
                    # Apply new mask
                    module.weight_mask = new_mask
        else:
            W_shapes = []
            res = []

            nnz = 0
            for name, module in self.named_modules():
                if 'weight_orig' not in module._parameters.keys():
                    continue
                mask = module.weight_mask
                weight = (module.weight_mask*module.weight_orig).abs()
                total_num = mask.numel()
                valid_num = int(mask.sum())
                zero_num = total_num-valid_num + int(percent*valid_num/100)

                W_shapes.append((name, weight.data.shape))
                res.append(weight.data.view(-1))
                nnz += zero_num
            res = torch.cat(res, dim=0)
            _, idx = torch.topk(res, nnz, largest=False, sorted=False)

            new_res = to_tensor(torch.ones_like(
                res), dtype='float', device=res.device)
            new_res[idx] = 0.0
            # # flattened array of nonzero values
            # percentile_value = percentile(param, (1-k)*100)
            # # Convert Tensors to numpy and calculate
            # param.data = torch.where(
            #     param < percentile_value, to_tensor([0.0]), param.data)
            offset = 0
            W_shapes = iter(W_shapes)
            for name, module in self.named_modules():
                if 'weight_orig' not in module._parameters.keys():
                    continue
                name_, shape = next(W_shapes)
                assert name_ == name
                mask = module.weight_mask
                mask.data = new_res[offset:offset+mask.numel()].view(shape)
                offset += mask.numel()

    def make_mask(self):
        for module in self.modules():
            if 'weight' in module._parameters.keys():
                prune.identity(module, 'weight')

    # Function for Initialization
    @staticmethod
    def weight_init(m):
        '''
        Usage:
            model = Model()
            model.apply(weight_init)
        '''
        if isinstance(m, nn.Conv1d):
            init.normal_(m.weight.data)
            if m.bias is not None:
                init.normal_(m.bias.data)
        elif isinstance(m, nn.Conv2d):
            init.xavier_normal_(m.weight.data)
            if m.bias is not None:
                init.normal_(m.bias.data)
        elif isinstance(m, nn.Conv3d):
            init.xavier_normal_(m.weight.data)
            if m.bias is not None:
                init.normal_(m.bias.data)
        elif isinstance(m, nn.ConvTranspose1d):
            init.normal_(m.weight.data)
            if m.bias is not None:
                init.normal_(m.bias.data)
        elif isinstance(m, nn.ConvTranspose2d):
            init.xavier_normal_(m.weight.data)
            if m.bias is not None:
                init.normal_(m.bias.data)
        elif isinstance(m, nn.ConvTranspose3d):
            init.xavier_normal_(m.weight.data)
            if m.bias is not None:
                init.normal_(m.bias.data)
        elif isinstance(m, nn.BatchNorm1d):
            init.normal_(m.weight.data, mean=1, std=0.02)
            init.constant_(m.bias.data, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.normal_(m.weight.data, mean=1, std=0.02)
            init.constant_(m.bias.data, 0)
        elif isinstance(m, nn.BatchNorm3d):
            init.normal_(m.weight.data, mean=1, std=0.02)
            init.constant_(m.bias.data, 0)
        elif isinstance(m, nn.Linear):
            init.xavier_normal_(m.weight.data)
            init.normal_(m.bias.data)
        elif isinstance(m, nn.LSTM):
            for param in m.parameters():
                if len(param.shape) >= 2:
                    init.orthogonal_(param.data)
                else:
                    init.normal_(param.data)
        elif isinstance(m, nn.LSTMCell):
            for param in m.parameters():
                if len(param.shape) >= 2:
                    init.orthogonal_(param.data)
                else:
                    init.normal_(param.data)
        elif isinstance(m, nn.GRU):
            for param in m.parameters():
                if len(param.shape) >= 2:
                    init.orthogonal_(param.data)
                else:
                    init.normal_(param.data)
        elif isinstance(m, nn.GRUCell):
            for param in m.parameters():
                if len(param.shape) >= 2:
                    init.orthogonal_(param.data)
                else:
                    init.normal_(param.data)

    def prune_test(self, percent=10, _iter=None, iter_prune=35, iter_train=100, _global=True, adv_train=None, reinit=False, _continue=False,  **kwargs):
        if not _continue:
            self.make_mask()
        prefix = '_prune'
        if adv_train is not None:
            prefix += '_adv_' + adv_train
        if _global:
            prefix += '_global'
        if _iter is None:
            for i in range(iter_prune):
                prefix = '_prune_%d' % i
                self.load_pretrained_weights(prefix=prefix)
                if reinit:
                    self.apply(self.weight_init)
                self.adv_train(iter_train, save=False, **kwargs)
        else:
            prefix = '_prune_%d' % _iter
            self.load_pretrained_weights(prefix=prefix)
            if reinit:
                self.apply(self.weight_init)
            self.adv_train(iter_train, save=False, **kwargs)

    def prune_atmc(self, epoch, perturb=None, percent=0.1, m=8, alpha=2.0/255, epsilon=8.0/255, iteration=20, lr=5e-3, train_opt='full', optim_type='SGD', lr_scheduler=True, validate_interval=10, save=True, prefix='_atmc', parallel=True, **kwargs):
        self.train()
        optimizer = self.define_optimizer(
            train_opt=train_opt, optim_type=optim_type, lr_scheduler=lr_scheduler, lr=lr, **kwargs)
        _lr_scheduler = None
        if lr_scheduler:
            _lr_scheduler = optimizer
            optimizer = _lr_scheduler.optimizer
        optimizer.zero_grad()

        if perturb is None:
            from package.utils.main_utils import get_perturb
            perturb = get_perturb(
                'pgd', model=self, iteration=iteration, alpha=alpha, epsilon=epsilon)

        _, best_acc, _ = self._validate()
        _, best_adv_acc, _ = self.adv_validate(perturb=perturb, targeted=False)
        # _, best_adv_acc, _ = self.adv_validate(
        #     validloader=validloader, perturb=perturb, targeted=True)
        self.train()

        losses = AverageMeter('Loss', ':.4e')
        top1 = AverageMeter('Acc@1', ':6.2f')
        top5 = AverageMeter('Acc@5', ':6.2f')

        for _epoch in range(epoch):
            losses.reset()
            top1.reset()
            top5.reset()
            for i, data in enumerate(self.dataset.loader['train']):
                # data_time.update(time.time() - end)
                _input, _label = self.get_data(data, mode='train')
                noise = to_tensor(torch.zeros_like(_input))
                for k in range(m):

                    X = to_valid_img(_input+noise).detach()
                    X.requires_grad = True
                    _output = self.get_logits(X, parallel=parallel)
                    loss = self.criterion(_output, _label)

                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                    self.project(k=percent)

                    noise = to_valid_img(
                        noise + epsilon * torch.sign(X.grad), - epsilon, epsilon).detach()

                    acc1, acc5 = self.accuracy(_output, _label, topk=(1, 5))
                    losses.update(loss.item(), _label.size(0))
                    top1.update(acc1[0], _label.size(0))
                    top5.update(acc5[0], _label.size(0))

                    # batch_time.update(time.time() - end)
                    # end = time.time()

                    # if i % 10 == 0:
                    #     progress.display(i)
            print(('Epoch: [%d/%d],' % (_epoch+1, epoch)).ljust(25, ' ') +
                  'Loss: %.4f,\tTop1 Acc: %.3f,\tTop5 Acc: %.3f' % (losses.avg, top1.avg, top5.avg))
            if lr_scheduler:
                _lr_scheduler.step()

            if validate_interval != 0:
                if (_epoch+1) % validate_interval == 0 or _epoch == epoch - 1:
                    _, cur_acc, _ = self._validate()
                    _, cur_adv_acc, _ = self.adv_validate(
                        perturb=perturb, targeted=False)
                    # _, cur_adv_acc, _ = self.adv_validate(perturb=perturb, targeted=True)
                    self.train()
                    if cur_adv_acc > best_adv_acc and save:
                        self.save_weights(prefix=prefix)
                        best_adv_acc = cur_adv_acc
                        print('current model saved!')
                    print('---------------------------------------------------')
        self.zero_grad()
        self.eval()

    # Input:
    # gamma (stepsize sequence)
    # n, T (update steps)
    # pho, k, b, Delta (hyper-parameters)
    # def prune_atmc(self, epoch=150, iteration=7, rho=0.1, k=0.1, Delta=4.0/255, perturb=None, bias=False, validate_interval=10, save=True, **kwargs):
        # lr_scheduler = self.define_optimizer(
        #     train_opt='full', optim_type='SGD', lr_scheduler=True)
        # optimizer = lr_scheduler.optimizer
        # optimizer.zero_grad()

        # alpha = 1.25 * Delta / iteration
        # if perturb is None:
        #     from package.utils.main_utils import get_perturb
        #     perturb = get_perturb(
        #         'pgd', model=self, iteration=iteration, alpha=alpha, epsilon=Delta)

        # # # Initialize theta_prime and u
        # # u = OrderedDict()
        # # theta_prime = OrderedDict()
        # # for name, param in self.named_parameters():
        # #     theta_prime[name] = deepcopy(param)
        # #     u[name] = to_tensor(torch.zeros_like(param))

        # self.train()
        # losses = AverageMeter('Loss', ':.4e')
        # top1 = AverageMeter('Acc@1', ':6.2f')
        # top5 = AverageMeter('Acc@5', ':6.2f')
        # for _epoch in range(epoch):
        #     losses.reset()
        #     top1.reset()
        #     top5.reset()
        #     for i, data in enumerate(self.dataset.loader['train']):
        #         _input, _label = self.get_data(data, mode='valid')
        #         adv_input, _iter = perturb.perturb(_input, targeted=False)
        #         _output = self.get_logits(adv_input)
        #         loss = self.criterion(_output, _label)
        #         loss.backward()
        #         optimizer.step()
        #         optimizer.zero_grad()
        #         self.project(k=k, bias=bias)

        #         acc1, acc5 = self.accuracy(_output, _label, topk=(1, 5))
        #         losses.update(loss.item(), _label.size(0))
        #         top1.update(acc1[0], _label.size(0))
        #         top5.update(acc5[0], _label.size(0))

        #     print(('Epoch: [%d/%d],' % (_epoch+1, epoch)).ljust(25, ' ') +
        #           'Loss: %.4f,\tTop1 Acc: %.3f,\tTop5 Acc: %.3f' % (losses.avg, top1.avg, top5.avg))
        #     lr_scheduler.step()
        #     # # update theta
        #     # for name, param in self.named_parameters():
        #     #     norm_grad = param.data-theta_prime[name]+u[name]
        #     #     param.data = param.data - \
        #     #         gamma[t]*(param.grad.data+rho*norm_grad)

        #     # # update theta_prime
        #     # zero_kmeans(add_dict(self.state_dict(), u), B=2**b)
        #     # # update u
        #     # for name, param in self.named_parameters():
        #     #     u[name] = u[name]+param.data-theta_prime[name]
        #     if validate_interval != 0:
        #         if (_epoch+1) % validate_interval == 0 or _epoch == epoch - 1:
        #             _, cur_acc, _ = self._validate()
        #             _, cur_adv_acc, _ = self.adv_validate(
        #                 perturb=perturb, targeted=False)
        #             # _, cur_adv_acc, _ = self.adv_validate(perturb=perturb, targeted=True)
        #             self.train()
        #             if cur_adv_acc > best_adv_acc and save:
        #                 self.save_weights(prefix=prefix)
        #                 best_adv_acc = cur_adv_acc
        #                 print('current model saved!')
        #             print('---------------------------------------------------')
        # self.zero_grad()
        # self.eval()

    def project(self, k=0.1, bias=False):
        W_shapes = []
        res = []
        for name, param in self.named_parameters():
            if bias:
                if 'weight' not in name and 'bias' not in name:
                    continue
            elif 'weight' not in name:
                continue
            W_shapes.append((name, param.data.shape))
            res.append(param.data.view(-1))
        res = torch.cat(res, dim=0)
        nnz = round(res.shape[0]*k)
        _, idx = torch.topk(res.abs(), int(
            res.shape[0]-nnz), largest=False, sorted=False)
        res[idx] = 0.0
        # # flattened array of nonzero values
        # percentile_value = percentile(param, (1-k)*100)
        # # Convert Tensors to numpy and calculate
        # param.data = torch.where(
        #     param < percentile_value, to_tensor([0.0]), param.data)
        offset = 0
        W_shapes = iter(W_shapes)
        for name, param in self.named_parameters():
            if bias:
                if 'weight' not in name and 'bias' not in name:
                    continue
            elif 'weight' not in name:
                continue
            _name, shape = next(W_shapes)
            assert _name == name
            param.data.copy_(res[offset:offset+param.numel()].view(shape))
            offset += param.numel()


# Input: U_bar (a set of real numbers), B (number of clusters)
# Output: U (quantized tensor)
# def zero_kmeans(U_bar, B):
#     # Initialize by randomly picked nonzero elements from U_bar
#     Q = [0.0]

#     return U


# def add_dict(dict1, dict2):
#     _dict = OrderedDict()
#     for key in dict1.keys():
#         _dict[key] = dict1[key]+dict2[key]
#     return _dict


# def minus_dict(dict1, dict2):
#     _dict = OrderedDict()
#     for key in dict1.keys():
#         _dict[key] = dict1[key]-dict2[key]
#     return _dict
