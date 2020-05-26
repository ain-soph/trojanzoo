# -*- coding: utf-8 -*-

from imports.universal import *

from utils.utils import *
from utils.model_utils import *
from model.model import *

from perturb.pgd import PGD

import argparse


class LID(Model):
    """docstring for Model"""

    def __init__(self, k, _model, name='lid', data_dir='./data/', dataset='cifar10', num_classes=2, **kwargs):
        num_classes = 2
        self.layer_num = _model.get_layer_num()['features'] + \
            _model.get_layer_num()['classifier']
        super(LID, self).__init__(name=name, data_dir=data_dir, dataset=dataset,
                                  num_classes=num_classes, conv_dim=self.layer_num, fc_depth=1, fc_dim=2, **kwargs)
        self.k = k
        self._model = _model
        self.features = LID_Features(k, _model, layer_num=self.layer_num)

        parser = argparse.ArgumentParser()
        parser.add_argument('-d', '--dataset',
                            dest='dataset', default=dataset)
        parser.add_argument('-i', '--input_method',
                            dest='input_method', default='pgd')
        parser.add_argument('--mode',
                            dest='mode', default='white')
        parser.add_argument('--data_dir', dest='data_dir', default='./data/')
        parser.add_argument(
            '--model_name', dest='model_name', default='resnetcomp18')
        parser.add_argument('--layer', dest='layer', default=None)
        args = parser.parse_args('')

        self._perturb = PGD(args)
        self._perturb.iteration = 20
        self._perturb.alpha = 2.0/255
        self._perturb.epsilon = 8.0/255
        self._perturb.untarget = True

    def forward(self, x, x_norm=None):
        x = self.features(x, x_norm=x_norm)
        x = self.get_logits_from_fm(x)
        return x

    def _train(self, trainloader, validloader, _model=None, model=None, epoch=100, lr=None):
        if _model is None:
            _model = self._model
        model = parallel_model(_model, model)
        model.train()
        optimizer = self.define_optimizer(lr=lr, train_opt='partial')

        train_set_pos = '/data/rbp5354/result/lid/%s/'% self.dataset
        if not os.path.exists(train_set_pos):
            os.makedirs(train_set_pos)
        train_set_pos += '%s_train.npy' % self._model.name
        if not os.path.exists(train_set_pos):
            print('start to extract training LID features')
            train_set = {}
            for i, (X, Y) in enumerate(trainloader):
                print('batch: ', i)
                X = to_tensor(X)
                X_noise = add_noise(X, std=8.0/255)
                model.eval()
                X_adv, _ = self._perturb.perturb(
                    _model, X, target=Y, model=model)
                model.train()
                _label = to_tensor(
                    torch.cat((torch.zeros(len(X)+len(X_noise)), torch.ones(len(X_adv)))), dtype='long')
                _features = torch.cat((self.features(X, x_norm=X), self.features(
                    X_noise, x_norm=X), self.features(X_adv, x_norm=X)))
                train_set[i] = {'X': X, 'X_noise': X_noise, 'X_adv': X_adv, '_label': _label, '_features': _features}
            np.save(train_set_pos, train_set)
            print('training set saved')
        else:
            train_set = np.load(train_set_pos, allow_pickle=True).item()
        for _epoch in range(epoch):
            losses = AverageMeter()
            top1 = AverageMeter()
            for i in train_set.keys():
                X = to_tensor(train_set[i]['X'])
                X_noise = to_tensor(train_set[i]['X_noise'])
                X_adv = to_tensor(train_set[i]['X_adv'])
                _label = to_tensor(train_set[i]['_label'])
                _features = to_tensor(train_set[i]['_features'])

                _output = self.get_logits_from_fm(_features)
                optimizer.zero_grad()  # zero the gradients buffer, reset the gradient in each update
                loss = self.criterion(_output, _label)
                loss.backward()
                optimizer.step()

                prec = accuracy(_output, _label)
                losses.update(loss.item(), _label.size(0))
                top1.update(prec[0], _label.size(0))
            print('Epoch: [%d/%d], Loss: %.4f, Acc: %.4f' %
                  (_epoch+1, epoch, losses.avg, top1.avg))
            if _epoch % 5 == 0:
                self._validate(validloader, _model, model, output=True)
                self.save_weights()
                print('current model saved!')
        model.eval()


    def _validate(self, validloader, _model=None, model=None, output=False):
        if _model is None:
            _model = self._model
        model = parallel_model(_model, model)
        model.eval()

        valid_set_pos = '/data/rbp5354/result/lid/%s/'% self.dataset
        if not os.path.exists(valid_set_pos):
            os.makedirs(valid_set_pos)
        valid_set_pos += '%s_valid.npy' % self._model.name
        if not os.path.exists(valid_set_pos):
            print('start to extract validate LID features')
            valid_set = {}
            for i, (X, Y) in enumerate(validloader):
                print('batch: ', i)
                X = to_tensor(X).detach()
                X_noise = add_noise(X, std=8.0/255).detach()
                model.eval()
                X_adv, _ = self._perturb.perturb(
                    _model, X, target=Y, model=model)
                model.train()
                _label = to_tensor(torch.ones(len(X_adv)), dtype='long') # torch.zeros(len(X)+len(X_noise)), 
                    # self.features(X, x_norm=X), self.features(
                    # X_noise, x_norm=X), 
                _features = self.features(X_adv, x_norm=X)
                valid_set[i] = {'X': X, 'X_noise': X_noise,
                                'X_adv': X_adv, '_label': _label, '_features': _features}
            np.save(valid_set_pos, valid_set)
            print('valid set saved')
        else:
            valid_set = np.load(valid_set_pos, allow_pickle=True).item()

        losses = AverageMeter()
        top1 = AverageMeter()

        for i in valid_set.keys():
            X = to_tensor(valid_set[i]['X'])
            X_noise = to_tensor(valid_set[i]['X_noise'])
            X_adv = to_tensor(valid_set[i]['X_adv'])
            _label = to_tensor(valid_set[i]['_label'])
            _features = to_tensor(valid_set[i]['_features'])

            _output = self.get_logits_from_fm(_features)
            loss = self.criterion(_output, _label)

            prec = accuracy(_output, _label)
            losses.update(loss.item(), _label.size(0))
            top1.update(prec[0], _label.size(0))
        if output:
            print('Loss: %.4f,\tTop1 Acc: %.3f' % (losses.avg, top1.avg))
        return losses.avg, top1.avg


class LID_Features(nn.Module):
    def __init__(self, k, _model, layer_num=None):
        super(LID_Features, self).__init__()
        self.k = k
        self._model = _model
        self.layer_num = layer_num if layer_num is not None else self._model.layer_num[
            'features']+self._model.layer_num['classifier']

    def forward(self, x, x_norm=None):
        with torch.no_grad():
            a = [self._model.get_layer(x, layer=l) for l in range(self.layer_num)]
            if x_norm is None:
                x_norm = x
                a_norm = a
            else:
                a_norm = [self._model.get_layer(x_norm, layer=l)
                        for l in range(self.layer_num)]
            result = []
            for l in range(self.layer_num):
                shape = list(a[l].shape)
                shape.insert(1, 1)
                shape_norm = list(a_norm[l].shape)
                shape_norm.insert(0, 1)
                y = (a[l].reshape(shape) - a_norm[l].reshape(shape_norm)
                    ).view(shape[0], shape[0], -1)
                y = y.norm(p=2, dim=-1)
                # .contiguous()
                y = y.topk(self.k+1, largest=False)[0][:, 1:]
                y = -1 / torch.log(y / y[:, -1, None]).mean(dim=-1)
                result.append(y)
            result = to_tensor(result).transpose(0, 1)
            return result
