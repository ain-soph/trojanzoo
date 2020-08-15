# -*- coding: utf-8 -*-

from trojanzoo.attack import Attack

from trojanzoo.utils.output import prints

import torch


class Poison_Basic(Attack):

    name: str = 'poison_basic'

    def __init__(self, percent: float = 0.01, **kwargs):
        super().__init__(**kwargs)
        self.param_list['poison'] = ['percent', 'target_idx']
        self.percent: float = percent
        self.target_idx: int = target_idx

        _, clean_acc, _ = self.model._validate(print_prefix='Baseline Clean',
                                               get_data=None, **kwargs)
        self.clean_acc = clean_acc
        self.temp_input: torch.Tensor = None
        self.temp_label: torch.LongTensor = None

    def attack(self, **kwargs):
        # model._validate()
        correct = 0
        total = 0
        total_iter = 0
        for data in self.dataset.loader['test']:
            if total >= 100:
                break
            _input, _label = self.model.remove_misclassify(data)
            if len(_label) == 0:
                continue
            self.model.load()
            adv_input, _iter = self._train(**kwargs)
            _, target_acc, clean_acc = self.validate_func()

            total += 1
            if _iter:
                correct += 1
                total_iter += _iter
            print(f'{correct} / {total}')
            print('current iter: ', _iter)
            print('succ rate: ', float(correct) / total)
            if correct > 0:
                print('avg  iter: ', float(total_iter) / correct)
            print('-------------------------------------------------')
            print()

    def _train(self, epoch: int, _input: torch.Tensor, _label: torch.LongTensor, save=False, **kwargs):
        def loss_fn(x: torch.Tensor, y: torch.LongTensor, **kwargs):
            return self.loss(x, y, **kwargs) + self.percent * self.loss(_input, _label)
        self.temp_input = _input
        self.temp_label = _label

        self.model._train(epoch=epoch, save=save,
                          loss_fn=loss_fn, save_fn=self.save,
                          validate_func=self.validate_func, **kwargs)

    def save(self, **kwargs):
        filename = self.get_filename(**kwargs)
        file_path = self.folder_path + filename
        self.model.save(file_path + '.pth')
        print('attack results saved at: ', file_path)

    def get_filename(self, **kwargs):
        return self.model.name

    def validate_func(self, indent: int = 0, verbose=True, **kwargs) -> (float, float, float):
        clean_loss = self.model.loss(self.temp_input)
        _output = self.model.get_logits(self.temp_input)
        target_acc, _ = self.model.accuracy(_output, self.temp_label, topk=(1, 5))

        clean_loss, clean_acc, _ = self.model._validate(print_prefix='Validate Clean',
                                                        indent=indent, verbose=verbose, **kwargs)
        if verbose:
            prints(f'Validate Target:       Loss: {clean_loss:10.4f}    Accuracy: {target_acc:7.3f}', indent=indent)
        # todo: Return value
        if self.clean_acc - clean_acc > 3 and self.clean_acc > 40:
            target_acc = 0.0
        return clean_loss + target_loss, target_acc, clean_acc
