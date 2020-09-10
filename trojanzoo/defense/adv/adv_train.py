# -*- coding: utf-8 -*-
# when pgd attack,
# use adv pgd.
# write a loss_fn to send into the pgd.

from ..defense import Defense
from trojanzoo.attack.adv import PGD

from torch import optim


class Adv_Train(Defense):

    name: str = 'adv_train'

    def __init__(self, pgd_alpha: float = 2 / 255, pgd_epsilon: float = 16 / 255, pgd_iteration = 20, **kwargs):
        super().__init__(**kwargs)
        self.pgd_alpha = pgd_alpha
        self.pgd_epsilon = pgd_epsilon
        self.pgd_iteration = pgd_iteration

        self.pgd = PGD(alpha=self.pgd_alpha, epsilon=self.pgd_epsilon, iteration=self.pgd_iteration,
                       target_idx=0, output=self.output, dataset=self.dataset, model=self.model)

    def detect(self, **kwargs):
        super().detect(**kwargs)
        # Implementation of free adversarial training
        # TODO: Initialize theta.
        delta = 0
        criterion = self.model.criterion
        optimizer = optim.Adam(self.model.parameters())
        for epoch in range(self.epochs):
            for i, (train_X, train_label) in enumerate(self.model.dataset.loader):
                adv_x, _ = self.pgd.craft_example(_input=train_X)
                optimizer.zero_grad()
                y_pred = self.model(adv_x)
                loss = loss_fn(y_pred, y)
                loss.backward()
                optimizer.step()

    def loss_fn(self):
