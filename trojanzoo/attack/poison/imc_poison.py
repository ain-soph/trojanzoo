# -*- coding: utf-8 -*-

from .basic import Poison_Basic

from trojanzoo.optim import PGD


class IMC_Poison(Poison_Basic):

    name: str = 'imc_poison'

    def __init__(self, pgd_alpha: float = 20 / 255, pgd_epsilon: float = 1.0, pgd_iteration: int = 20, **kwargs):
        super().__init__(**kwargs)
        self.pgd_alpha: float = pgd_alpha
        self.pgd_epsilon: float = pgd_epsilon
        self.pgd_iteration: int = pgd_iteration
        self.pgd = PGD(dataset=self.dataset, model=self.model,
                       alpha=self.pgd_alpha, epsilon=self.pgd_epsilon, iteration=self.pgd_iteration,
                       target_idx=self.target_idx, universal=True, output=0)

    def _train(self, epoch: int, _input: torch.Tensor, _label: torch.LongTensor, save=False, **kwargs):
        def loss_fn(x: torch.Tensor, y: torch.LongTensor, **kwargs):
            return self.loss(x, y, **kwargs) + self.percent * self.loss(_input, _label)
        kwargs['save'] = False
        self.model._train(epoch=epoch, save=save, loss_fn=loss_fn, save_fn=self.save, **kwargs)

    def save(self, **kwargs):
        filename = self.get_filename(**kwargs)
        file_path = self.folder_path + filename
        self.model.save(file_path + '.pth')
        print('attack results saved at: ', file_path)

    def get_filename(self, **kwargs):
        return self.model.name
