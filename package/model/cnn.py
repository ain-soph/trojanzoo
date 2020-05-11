# -*- coding: utf-8 -*-

import package
from package.imports.universal import *
from package.utils.utils import *
from package.utils.output import prints
from package.utils.model import split_name as func
from package.dataset.dataset import Dataset

from collections import OrderedDict
from tqdm import tqdm
import time
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.utils import _pair


class _CNN(nn.Module):
    def __init__(self, num_classes: int = None, conv_depth=0, conv_dim=0, fc_depth=0, fc_dim=0, **kwargs):
        super(_CNN, self).__init__()

        self.conv_depth = conv_depth
        self.conv_dim = conv_dim
        self.fc_depth = fc_depth
        self.fc_dim = fc_dim
        self.num_classes = num_classes

        self.features = self.define_features()   # feature extractor
        self.avgpool = nn.Identity()  # average pooling
        self.classifier = self.define_classifier()  # classifier

    # forward method
    # input: (batch_size, channels, height, width)
    # output: (batch_size, logits)

    def forward(self, x):
        # if x.shape is (channels, height, width)
        # (channels, height, width) ==> (batch_size: 1, channels, height, width)
        x = self.get_final_fm(x)
        x = self.get_logits_from_fm(x)
        return x

    # input: (batch_size, channels, height, width)
    # output: (batch_size, [feature_map])
    def get_fm(self, x):
        return self.features(x)

    # get feature map
    # input: (batch_size, channels, height, width)
    # output: (batch_size, [feature_map])
    def get_final_fm(self, x):
        x = self.get_fm(x)
        x = self.avgpool(x)
        x = x.flatten(start_dim=1)
        return x

    # get logits from feature map
    # input: (batch_size, [feature_map])
    # output: (batch_size, logits)
    def get_logits_from_fm(self, x):
        return self.classifier(x)

    def define_features(self, conv_depth: int = None, conv_dim: int = None):
        return nn.Identity()

    def define_classifier(self, num_classes: int = None, conv_dim: int = None, fc_depth: int = None, fc_dim: int = None):
        if fc_depth is None:
            fc_depth = self.fc_depth
        if self.fc_depth <= 0:
            return nn.Identity()
        if conv_dim is None:
            conv_dim = self.conv_dim
        if fc_dim is None:
            fc_dim = self.fc_dim
        if num_classes is None:
            num_classes = self.num_classes

        seq = []
        if self.fc_depth == 1:
            seq.append(('fc', nn.Linear(self.conv_dim, self.num_classes)))
        else:
            seq.append(('fc1', nn.Linear(self.conv_dim, self.fc_dim)))
            seq.append(('relu1', nn.ReLU()))
            seq.append(('dropout1', nn.Dropout()))
            for i in range(self.fc_depth-2):
                seq.append(
                    ('fc'+str(i+2), nn.Linear(self.fc_dim, self.fc_dim)))
                seq.append(('relu'+str(i+2), nn.ReLU()))
                seq.append(('dropout'+str(i+2), nn.Dropout()))
            seq.append(('fc'+str(self.fc_depth),
                        nn.Linear(self.fc_dim, self.num_classes)))
        return nn.Sequential(OrderedDict(seq))


class CNN(object):
    """docstring for CNN"""

    @staticmethod
    def split_name(*args, **kwargs):
        return func(*args, **kwargs)

    def __init__(self, name='cnn', dataset: Dataset = None,
                 num_classes: int = None, loss_weights: torch.FloatTensor = None, model_class=_CNN, get_data=None,
                 folder_path: str = None, pretrain=True, prefix='', cache_threshold=0.0, adv_train=False, **kwargs):
        self.name = name
        self.dataset = dataset
        self.prefix = prefix

        #------------Auto--------------#
        self.prefix += ('_adv_train' if adv_train else '')
        if dataset is not None:
            if isinstance(dataset, str):
                pass
            if folder_path is None:
                # Default Folder Path
                folder_path = dataset.data_dir+dataset.data_type+'/'+dataset.name+'/model/'
            if num_classes is None:
                num_classes = dataset.num_classes
            if loss_weights is None:
                loss_weights = dataset.loss_weights
        if num_classes is None:
            num_classes = 1000
        self.num_classes = num_classes  # number of classes
        #------------------------------#

        #---------Folder Path----------#
        if folder_path is None:
            folder_path = '~/'
        self.folder_path = folder_path
        if not os.path.exists(self.folder_path):
            os.makedirs(self.folder_path)
        #------------------------------#
        self.loss_weights = loss_weights
        self.criterion = self.define_criterion(loss_weights=loss_weights)
        self.softmax = nn.Softmax(dim=1)
        if get_data is None:
            if self.dataset is not None:
                get_data = self.dataset.get_data
            else:
                def _get_data(data): return data[0], data[1]
                get_data = _get_data
        self.get_data = get_data

        #-----------Temp---------------#
        # the location when loading pretrained weights using torch.load
        self.cuda_available = True if torch.cuda.is_available() else False
        self.device = 'cuda' if self.cuda_available else 'cpu'
        self.num_gpus = torch.cuda.device_count()
        self._model = model_class(
            num_classes=num_classes, model=self, **kwargs)
        self.cache_threshold = cache_threshold
        self.model = self.get_parallel()
        # load pretrained weights
        self.load_pretrained_weights(
            weights_loc='' if pretrain == True else 'nothing')
        if self.cuda_available:
            self.cuda()
        self.eval()

    #-----------------Operation on Data----------------------#

    def get_logits(self, _input, parallel=True, randomized_smooth=False, sigma=0.01, n=100, **kwargs):
        model = self.model if parallel else self._model
        if randomized_smooth:
            _list = []
            for _ in range(n):
                _input_noise = add_noise(_input, std=sigma, detach=False)
                _list.append(model(_input_noise))
            return torch.stack(_list).mean(dim=0)
        else:
            return model(_input)

    def get_prob(self, _input, **kwargs):
        return self.softmax(self.get_logits(_input, **kwargs))

    def get_class(self, _input, **kwargs):
        return self.get_logits(_input, **kwargs).argmax(dim=-1)

    def loss(self, _input, label, **kwargs):
        return self.criterion(self(_input, **kwargs), label)

    def remove_misclassify_from_batch(self, data):
        _input, _label = self.get_data(data)
        _classification = self.get_class(_input)

        repeat_idx = _classification.eq(_label)
        _input = _input[repeat_idx]
        _label = _label[repeat_idx]
        return _input, _label
    #--------------------------------------------------------#

    # Define the optimizer
    # and transfer to that tuning mode.
    # train_opt: 'full' or 'partial' (default: 'partial')
    # lr: (default: [full:2e-3, partial:2e-4])
    # optim_type: to be implemented
    #
    # return: optimizer

    def define_optimizer(self, train_opt='partial', lr: float = None, optim_type: str = None, lr_scheduler=False, **kwargs):
        if train_opt != 'full' and train_opt != 'partial':
            print('train_opt = %s' % train_opt)
            raise ValueError(
                'Value of Parameter "train_opt" shoule be "full" or "partial"! ')
        if optim_type is None:
            optim_class = optim.SGD if train_opt == 'full' else optim.Adam
        else:
            optim_class = getattr(optim, optim_type)
        if lr is None:
            lr = 0.01 if train_opt == 'full' else 1e-4
        parameters = self._model.parameters(
        ) if train_opt == 'full' else self._model.classifier.parameters()
        self.transfer_tuning(train_opt)

        if kwargs == {}:
            if optim_class == optim.SGD:
                print('using default SGD optimizer setting')
                kwargs = {'momentum': 0.9,
                          'weight_decay': 2e-4, 'nesterov': True}
        else:
            print('kwargs: ', kwargs)

        optimizer = optim_class(parameters, lr, **kwargs)
        if lr_scheduler:
            print('enable lr_scheduler')
            optimizer = optim.lr_scheduler.StepLR(
                optimizer, step_size=30, gamma=0.1)
            # optimizer = optim.lr_scheduler.MultiStepLR(
            #     optimizer, milestones=[150, 250], gamma=0.1)
        return optimizer

    # define loss function
    # Cross Entropy
    def define_criterion(self, loss_weights: torch.FloatTensor = None):
        return nn.CrossEntropyLoss(weight=loss_weights)

    #-----------------------------Load & Save Model-------------------------------------------#

    # weights_loc: (default: '') if '', use the default path. Else if the path doesn't exist, quit.
    # full: (default: False) whether save feature extractor.
    # output: (default: False) whether output help information.
    def load_pretrained_weights(self, weights_loc='', prefix: str = None, full=True, map_location='default', output=False):
        if prefix is None:
            prefix = self.prefix
        if map_location is not None:
            if map_location == 'default':
                map_location = 'cuda' if self.cuda_available else 'cpu'
        try:
            if weights_loc == '':
                weights_loc = self.folder_path + self.name + prefix + '.pth'
                if os.path.exists(weights_loc):
                    if output:
                        print("********Load From Saved File: %s********" %
                              weights_loc)
                    self._model.load_state_dict(torch.load(
                        weights_loc, map_location=map_location))
                else:
                    print('Default model file not exist: ', weights_loc)
                    self.load_official_weights()
            elif os.path.exists(weights_loc):
                if output:
                    print("********Load From Saved File: %s********" %
                          weights_loc)
                if full:
                    self._model.load_state_dict(
                        torch.load(weights_loc, map_location=map_location))
                else:
                    self._model.classifier.load_state_dict(
                        torch.load(weights_loc, map_location=map_location))
            else:
                print('File not Found: ', weights_loc)
                self.load_official_weights()
        except:
            print('weight_loc: ', weights_loc)
            raise ValueError()

    # weights_loc: (default: '') if '', use the default path.
    # full: (default: False) whether save feature extractor.
    def save_weights(self, weights_loc='', prefix: str = None, full=True):
        if prefix is None:
            prefix = self.prefix
        if weights_loc == '':
            weights_loc = self.folder_path+self.name+prefix+'.pth'
            torch.save(self._model.state_dict(), weights_loc)
        else:
            if full:
                torch.save(self._model.state_dict(), weights_loc)
            else:
                torch.save(self._model.classifier.state_dict(), weights_loc)

    # define in concrete model class.
    def load_official_weights(self, output=True):
        if output:
            print("Nothing Happens. No official pretrained data to load.")
    #-----------------------------------------------------------------------------------------#

    #-----------------------------------Train and Validate------------------------------------#
    def _train(self, epoch, train_opt='full', lr_scheduler=False,
               validate_interval=10, full=True, save=True, prefix: str = None, parallel=True,
               loader_train: torch.utils.data.DataLoader = None, loader_valid: torch.utils.data.DataLoader = None, **kwargs):
        self.train()
        optimizer = self.define_optimizer(
            train_opt=train_opt, lr_scheduler=lr_scheduler, **kwargs)
        _lr_scheduler = None
        if lr_scheduler:
            _lr_scheduler = optimizer
            optimizer = _lr_scheduler.optimizer
        optimizer.zero_grad()

        if loader_train is None:
            loader_train = self.dataset.loader['train']

        _, best_acc, _ = self._validate(loader=loader_valid)
        self.train()

        # batch_time = AverageMeter('Time', ':6.3f')
        # data_time = AverageMeter('Data', ':6.3f')
        losses = AverageMeter('Loss', ':.4e')
        top1 = AverageMeter('Acc@1', ':6.2f')
        top5 = AverageMeter('Acc@5', ':6.2f')
        # progress = ProgressMeter(
        #     len(trainloader),
        #     [batch_time, data_time, losses, top1, top5],
        #     prefix="Epoch: [{}]".format(epoch))

        end = time.time()
        for _epoch in range(epoch):
            losses.reset()
            top1.reset()
            top5.reset()
            for i, data in enumerate(loader_train):
                # data_time.update(time.time() - end)
                _input, _label = self.get_data(data, mode='train')
                _output = self.get_logits(
                    _input, parallel=parallel)
                loss = self.criterion(_output, _label)

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                acc1, acc5 = self.accuracy(_output, _label, topk=(1, 5))
                losses.update(loss.item(), _label.size(0))
                top1.update(acc1[0], _label.size(0))
                top5.update(acc5[0], _label.size(0))

                empty_cache(self.cache_threshold)

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
                    _, cur_acc, _ = self._validate(loader=loader_valid)
                    self.train()
                    if cur_acc > best_acc and save:
                        self.save_weights(prefix=prefix)
                        best_acc = cur_acc
                        print('current model saved!')
                    print('---------------------------------------------------')
        self.zero_grad()
        self.eval()

    def _validate(self, full=True, parallel=True, output=True, loader: torch.utils.data.DataLoader = None, indent=0, **kwargs):
        self.eval()
        # batch_time = AverageMeter('Time', ':6.3f')
        losses = AverageMeter('Loss', ':.4e')
        top1 = AverageMeter('Acc@1', ':6.2f')
        top5 = AverageMeter('Acc@5', ':6.2f')
        if loader is None:
            loader = self.dataset.loader['valid'] if full else self.dataset.loader['valid2']

        # progress = ProgressMeter(
        #     len(self.dataset.loader['valid']),
        #     [batch_time, losses, top1, top5],
        #     prefix='Test: ')
        with torch.no_grad():
            # end = time.time()

            for i, data in enumerate(loader):
                _input, _label = self.get_data(data, mode='valid')
                _output = self.get_logits(_input, parallel=parallel, **kwargs)
                loss = self.criterion(_output, _label)

                # measure accuracy and record loss
                acc1, acc5 = self.accuracy(_output, _label, topk=(1, 5))
                losses.update(loss.item(), _label.size(0))
                top1.update(acc1[0], _label.size(0))
                top5.update(acc5[0], _label.size(0))

                # empty_cache(self.cache_threshold)

                # measure elapsed time
                # batch_time.update(time.time() - end)
                # end = time.time()

                # if i % 10 == 0:
                #     progress.display(i)

            if output:
                prints('Validate'.ljust(25, ' ') +
                       'Loss: %.4f,\tTop1 Acc: %.3f, \tTop5 Acc: %.3f' % (losses.avg, top1.avg, top5.avg), indent=indent)
        return losses.avg, top1.avg, top5.avg
    #-----------------------------------------------------------------------------------------#

    #-----------------------------Adversarial Train and Validate------------------------------#
    def adv_train(self, epoch, mode='free', perturb = None,
                  m=8, alpha=2.0/255, epsilon=8.0/255, p=float('inf'), iteration=20, n=10, adre_lambda=0.1,
                  train_opt='full', lr_scheduler=False, validate_full=True,
                  validate_interval=10, save=True, prefix='_adv_train', parallel=True, **kwargs):
        if mode is None:
            self._train(epoch, train_opt=train_opt, lr_scheduler=lr_scheduler,
                        validate_interval=validate_interval, save=save, prefix=prefix, parallel=parallel, **kwargs)
            return

        optimizer = self.define_optimizer(
            train_opt=train_opt, lr_scheduler=lr_scheduler, **kwargs)
        _lr_scheduler = None
        if lr_scheduler:
            _lr_scheduler = optimizer
            optimizer = _lr_scheduler.optimizer
        optimizer.zero_grad()

        # if mode == 'adre':
        #     # p = 2
        #     # epsilon = 4.0
        #     # m=2
        #     alpha = 2*epsilon/m
        #     iteration = m
        if perturb is None:
            from package.utils.main_utils import get_perturb
            perturb = get_perturb(
                'pgd', model=self, iteration=iteration, alpha=alpha, epsilon=epsilon, targeted=False, p=p, early_stop=False)
        elif isinstance(perturb, str):
            from package.utils.main_utils import get_perturb
            perturb = get_perturb(
                perturb, model=self, iteration=iteration, alpha=alpha, epsilon=epsilon, targeted=False, p=p, early_stop=False)

        _, best_acc, _ = self._validate()
        _, best_adv_acc, _ = self.adv_validate(
            perturb=perturb, targeted=False, full=validate_full)
        # best_acc = 10
        # best_adv_acc = 10

        losses = AverageMeter('Loss', ':.4e')
        top1 = AverageMeter('Acc@1', ':6.2f')
        top5 = AverageMeter('Acc@5', ':6.2f')

        end = time.time()
        self.train()
        for _epoch in range(epoch):
            losses.reset()
            top1.reset()
            top5.reset()
            for i, data in enumerate(self.dataset.loader['train']):
                # data_time.update(time.time() - end)
                if i == 1 or i == 2:
                    continue
                _input, _label = self.get_data(data, mode='train')
                noise = to_tensor(torch.zeros_like(_input))
                adv_X = _input
                for k in range(m):
                    if mode == 'free':
                        _output = self.get_logits(
                            adv_X, parallel=parallel)
                        loss = self.criterion(_output, _label)
                        loss.backward()
                        optimizer.step()
                        optimizer.zero_grad()

                        acc1, acc5 = self.accuracy(
                            _output, _label, topk=(1, 5))
                        losses.update(loss.item(), _label.size(0))
                        top1.update(acc1[0], _label.size(0))
                        top5.update(acc5[0], _label.size(0))

                        empty_cache(self.cache_threshold)

                    if mode == 'adre':
                        _output = self.get_logits(
                            adv_X, parallel=parallel, randomized_smooth=True, n=n)
                        _, indices = _output.sort(dim=-1, descending=True)
                        idx0 = to_tensor(indices[:, 0])
                        idx1 = to_tensor(indices[:, 1])
                        target = torch.where(idx0 == _label, idx1, idx0)

                        loss = self.criterion(
                            _output, _label)-adre_lambda*self.loss(adv_X, target)
                        loss.backward()
                        optimizer.step()
                        optimizer.zero_grad()

                        acc1, acc5 = self.accuracy(
                            _output, _label, topk=(1, 5))
                        losses.update(loss.item(), _label.size(0))
                        top1.update(acc1[0], _label.size(0))
                        top5.update(acc5[0], _label.size(0))

                        empty_cache(self.cache_threshold)
                    self.eval()
                    adv_X, _ = perturb.perturb(_input, noise=noise, target=_label, targeted=False,
                                               iteration=1, randomized_smooth=(mode == 'adre'), n=n)
                    optimizer.zero_grad()
                    self.zero_grad()

                    self.train()
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
                    _, cur_acc, _ = self._validate(full=validate_full)
                    _, cur_adv_acc, _ = self.adv_validate(
                        perturb=perturb, targeted=False, full=validate_full)
                    # _, cur_adv_acc, _ = self.adv_validate(perturb=perturb, targeted=True)
                    self.train()
                    if cur_adv_acc > best_adv_acc and save:
                        self.save_weights(prefix=prefix)
                        best_adv_acc = cur_adv_acc
                        print('current model saved!')
                    print('---------------------------------------------------')
        self.zero_grad()
        self.eval()

    def adv_validate(self, perturb, targeted=False, full=True,
                     loader: torch.utils.data.DataLoader = None, parallel=True, output=True, **kwargs):
        self.eval()
        # batch_time = AverageMeter('Time', ':6.3f')
        losses = AverageMeter('Loss', ':.4e')
        top1 = AverageMeter('Acc@1', ':6.2f')
        top5 = AverageMeter('Acc@5', ':6.2f')

        if loader is None:
            loader = self.dataset.loader['valid'] if full else self.dataset.loader['valid2']
        # progress = ProgressMeter(
        #     len(self.dataset.loader['valid']),
        #     [batch_time, losses, top1, top5],
        #     prefix='Test: ')
        # end = time.time()
        total_num = 0
        for i, data in enumerate(self.dataset.loader['valid']):
            _input, _label = self.get_data(data, mode='valid')
            target = perturb.generate_target(_input) if targeted else _label
            if len(_label) == 0:
                continue
            adv_input, _iter = perturb.perturb(
                _input, target=target, targeted=targeted)

            with torch.no_grad():
                _output = self.get_logits(adv_input, parallel=parallel)
                loss = self.criterion(_output, _label)

                # measure accuracy and record loss
                acc1, acc5 = self.accuracy(_output, _label, topk=(1, 5))
                losses.update(loss.item(), _label.size(0))
                top1.update(acc1[0], _label.size(0))
                top5.update(acc5[0], _label.size(0))

                # empty_cache(self.cache_threshold)

                total_num += 1
                if total_num >= 100:
                    break

                # measure elapsed time
                # batch_time.update(time.time() - end)
                # end = time.time()

                # if i % 10 == 0:
                #     progress.display(i)

        if output:
            print(('Adv Validate '+('target' if targeted else 'untarget')).ljust(25, ' ') +
                  'Loss: %.4f,\tTop1 Acc: %.3f, \tTop5 Acc: %.3f' % (losses.avg, top1.avg, top5.avg))
        return losses.avg, top1.avg, top5.avg
    #-----------------------------------------------------------------------------------------#

    #-------------------------------------------Utility---------------------------------------#
    def accuracy(self, _output: torch.FloatTensor, _label: torch.LongTensor, topk=(1, 5)):
        """Computes the precision@k for the specified values of k"""
        with torch.no_grad():
            maxk = min(max(topk), self.num_classes)
            batch_size = _label.size(0)

            _, pred = _output.topk(maxk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(_label.view(1, -1).expand_as(pred))

            res = []
            for k in topk:
                if k > self.num_classes:
                    res.append(to_tensor([100.0]))
                else:
                    correct_k = correct[:k].view(-1).float().sum(0,
                                                                 keepdim=True)
                    res.append(correct_k.mul_(100.0 / batch_size))
            return res

    def transfer_tuning(self, train_opt: str = 'partial'):
        # make parameters of classifier trainable and freeze the feature extractor
        if train_opt == 'partial':
            for param in self._model.classifier.parameters():
                param.requires_grad = True
            for param in self._model.features.parameters():
                param.requires_grad = False
        # make parameters of the whole model trainable
        if train_opt == 'full':
            for param in self._model.parameters():
                param.requires_grad = True

    def get_parallel(self):
        if self.cuda_available and self.num_gpus > 1:
            if self.name[0] == 'g' and self.name[2] == 'n':
                return self._model
            else:
                return nn.DataParallel(self._model).cuda()
        else:
            return self._model

    @staticmethod
    def output_layer_information(layer, verbose=0, indent=0):
        for name, module in layer.named_children():
            prints(name.ljust(50-indent, ' '), indent=indent, end='')
            _str = ''
            if verbose > 0:
                _str = str(module).split('\n')[0]
                if _str[-1] == '(':
                    _str = _str[:-1]
            print(_str)
            CNN.output_layer_information(
                module, verbose=verbose, indent=indent+10)

    def summary(self, **kwargs):
        self.output_layer_information(self._model, **kwargs)

        #-----------------------------------------------------------------------------------------#

        #-----------------------------------------Reload------------------------------------------#
    def __call__(self, *args, **kwargs):
        return self.get_logits(*args, **kwargs)

    def train(self, mode=True):
        self._model.train(mode=mode)
        self.model.train(mode=mode)
        return self

    def eval(self):
        self._model.eval()
        self.model.eval()
        return self

    def cuda(self, device=None):
        self._model.cuda(device=device)
        self.model.cuda(device=device)
        return self

    def zero_grad(self):
        self._model.zero_grad()

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        return self._model.state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)

    def load_state_dict(self, state_dict, strict=True):
        return self._model.load_state_dict(state_dict, strict=strict)

    def parameters(self, recurse=True):
        return self._model.parameters(recurse=recurse)

    def named_parameters(self, prefix='', recurse=True):
        return self._model.named_parameters(prefix=prefix, recurse=recurse)

    def children(self):
        return self._model.children()

    def named_children(self):
        return self._model.named_children()

    def modules(self):
        return self._model.modules()

    def named_modules(self, memo=None, prefix=''):
        return self._model.named_modules(memo=memo, prefix=prefix)

    def apply(self, fn):
        return self._model.apply(fn)

    #-----------------------------------------------------------------------------------------#
    def generate_target(self, _input, idx=1, same=False):
        if len(_input.shape) == 3:
            _input = _input.unsqueeze(0)
        self.batch_size = _input.shape[0]
        _output = self.get_logits(_input)
        _, indices = _output.sort(dim=-1, descending=True)
        target = to_tensor(indices[:, idx])
        if same:
            target = repeat_to_batch(target.mode(dim=0)[0], len(_input))
        return target

    def get_target_confidence(self, _input, target, **kwargs):
        _result = self.get_prob(_input, **kwargs)
        if len(_result) == 1:
            return float(_result[0][int(target)])
        else:
            return np.array([float(_result[i][int(target[i])]) for i in range(len(_result))])


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=''):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def conv2d_same_padding(input, weight, bias=None, stride=1, padding=1, dilation=1, groups=1):
    """
    Conv2d layer with padding=same
    the padding param here is not important
    """

    input_rows = input.size(2)
    filter_rows = weight.size(2)
    effective_filter_size_rows = (filter_rows - 1) * dilation[0] + 1
    out_rows = (input_rows + stride[0] - 1) // stride[0]
    padding_rows = max(0, (out_rows - 1) * stride[0] +
                       (filter_rows - 1) * dilation[0] + 1 - input_rows)
    rows_odd = (padding_rows % 2 != 0)
    padding_cols = max(0, (out_rows - 1) * stride[0] +
                       (filter_rows - 1) * dilation[0] + 1 - input_rows)
    cols_odd = (padding_rows % 2 != 0)

    if rows_odd or cols_odd:
        input = F.pad(input, [0, int(cols_odd), 0, int(rows_odd)])

    return F.conv2d(input, weight, bias, stride[0],
                    padding=(padding_rows // 2, padding_cols // 2),
                    dilation=dilation, groups=groups)


class Conv2d_SAME(_ConvNd):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros'):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(Conv2d_SAME, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _pair(0), groups, bias, padding_mode)

    def forward(self, input):
        return conv2d_same_padding(input, self.weight, self.bias, self.stride,
                                   self.padding, self.dilation, self.groups)
