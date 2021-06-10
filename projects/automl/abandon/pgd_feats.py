#!/usr/bin/env python3


from typing import Union
from collections.abc import Callable
from trojanvision.attacks import PGD
from trojanzoo.utils.output import prints
from trojanzoo.utils import to_list

import torch
import trojanvision.environ
import trojanvision.datasets
import trojanvision.models
import trojanvision.trainer
import trojanvision.attacks

from trojanvision.utils import summary
import argparse

import warnings
warnings.filterwarnings("ignore")


class PGD_Feats(PGD):
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
            adv_input, _iter = self.craft_example(_input, **kwargs)

            total += 1
            if _iter:
                correct += 1
                total_iter += _iter
            else:
                total_iter += self.iteration
            print(f'{correct} / {total}')
            print('current iter: ', _iter)
            print('succ rate: ', float(correct) / total)
            print('avg  iter: ', float(total_iter) / total)
            print('-------------------------------------------------')
            print()

    def craft_example(self, _input: torch.Tensor, loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = None,
                      target: Union[torch.Tensor, int] = None, target_idx: int = None, **kwargs) -> tuple[torch.Tensor, int]:
        if len(_input) == 0:
            return _input, None
        if target_idx is None:
            target_idx = self.target_idx
        _feats = self.model.get_final_fm(_input)
        if loss_fn is None and self.loss_fn is None:
            if target is None:
                target = self.generate_target(_input, idx=target_idx)
            elif isinstance(target, int):
                target = target * torch.ones(len(_input), dtype=torch.long, device=_input.device)

            def _loss_fn(_feats: torch.Tensor, **kwargs):
                t = target
                if len(_feats) != len(target) and len(target) == 1:
                    t = target * torch.ones(len(_feats), dtype=torch.long, device=_feats.device)
                loss = self.model.loss(_label=t, _output=self.model._model.classifier(_feats), **kwargs)
                return loss if target_idx else -loss
            loss_fn = _loss_fn
        return self.optimize(_feats, loss_fn=loss_fn, target=target, **kwargs)

    def early_stop_check(self, X: torch.Tensor, target: torch.Tensor = None, loss_fn=None, **kwargs):
        if not self.stop_threshold:
            return False
        with torch.no_grad():
            _prob: torch.Tensor = self.model._model.softmax(
                self.model._model.classifier(X))
            _confidence = _prob.gather(dim=1, index=target.unsqueeze(1)).flatten()
        if self.target_idx and _confidence.min() > self.stop_threshold:
            return True
        if not self.target_idx and _confidence.max() < self.stop_threshold:
            return True
        return False

    def output_info(self, _input: torch.Tensor, noise: torch.Tensor, target: torch.Tensor, **kwargs):
        super(PGD, self).output_info(_input, noise, **kwargs)
        # prints('Original class     : ', to_list(_label), indent=self.indent)
        # prints('Original confidence: ', to_list(_confidence), indent=self.indent)
        with torch.no_grad():
            _prob: torch.Tensor = self.model._model.softmax(
                self.model._model.classifier(_input + noise))
            _confidence = _prob.gather(dim=1, index=target.unsqueeze(1)).flatten()
        prints('Target   class     : ', to_list(target), indent=self.indent)
        prints('Target   confidence: ', to_list(_confidence), indent=self.indent)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    trojanvision.environ.add_argument(parser)
    trojanvision.datasets.add_argument(parser)
    trojanvision.models.add_argument(parser)
    trojanvision.trainer.add_argument(parser)
    trojanvision.attacks.add_argument(parser)
    args = parser.parse_args()

    env = trojanvision.environ.create(**args.__dict__)
    dataset = trojanvision.datasets.create(**args.__dict__)
    model = trojanvision.models.create(dataset=dataset, **args.__dict__)
    trainer = trojanvision.trainer.create(dataset=dataset, model=model, **args.__dict__)
    attack = trojanvision.attacks.create(dataset=dataset, model=model, **args.__dict__, class_dict={'pgd': PGD_Feats})

    if env['verbose']:
        summary(env=env, dataset=dataset, model=model, train=trainer, attack=attack)
    attack.attack(**trainer)
