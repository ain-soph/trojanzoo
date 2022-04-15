#!/usr/bin/env python3

from ...abstract import BackdoorAttack
from trojanzoo.utils.model import init_weights
from trojanvision.attacks.adv import PGD    # TODO: Need to check whether this will cause ImportError
from trojanvision.optim import PGDoptimizer
from trojanvision.environ import env
from trojanzoo.utils.data import TensorListDataset, dataset_to_tensor, sample_batch

import torch
import torch.nn as nn
import torch.optim as optim

import argparse
import os

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    import torch.utils.data


class LabelConsistent(BackdoorAttack):
    r"""
    Contributor: Xiangshan Gao, Ren Pang

    Clean Label Backdoor Attack is described in detail in the paper `Clean Label Backdoor Attack`_ by Alexander Turner.

    The main idea is to perturb the poisoned samples
    in order to render learning the salient characteristic of the input more difficult,
    causing the model rely more heavily on the backdoor pattern in order to successfully introduce backdoor.

    Utilize the adversarial examples and GAB generated data,
    the resulted poisoned inputs appear to be consistent with their label
    and thus seem benign even upon human inspection.

    The authors haven't posted `original source code`_.

    Args:
        poison_generation_method (str): the chosen method to generate poisoned sample. Default: 'pgd'.
        tau (float): the interpolation constant used to balance source imgs and target imgs. Default: 0.4.
        epsilon (float): the perturbation bound in input space. Default: 0.1.
        noise_dim (int): the dimension of the input in the generator. Default: 100.
        generator_iters (int): the epochs for training the generator. Default: 1000.
        critic_iter (int): the critic iterations per generator training iteration. Default: 5.


    .. _Clean Label:
        https://people.csail.mit.edu/madry/lab/cleanlabel.pdf

    .. _related code:
        https://github.com/igul222/improved_wgan_training
        https://github.com/MadryLab/cifar10_challenge
        https://github.com/caogang/wgan-gp

    """
    name: str = 'label_consistent'

    @classmethod
    def add_argument(cls, group: argparse._ArgumentGroup):
        super().add_argument(group)
        group.add_argument('--poison_generation_method', choices=['pgd', 'gan'],
                           help='the chosen method to generate poisoned sample '
                           '(default: "pgd")')
        group.add_argument('--pgd_alpha', type=float)
        group.add_argument('--pgd_eps', type=float)
        group.add_argument('--pgd_iter', type=int)
        group.add_argument('--tau', type=float,
                           help='the interpolation constant used to balance source imgs and target imgs, '
                           'defaults to 0.2')
        group.add_argument('--noise_dim', type=int,
                           help='the dimension of the input in the generator, '
                           'defaults to config[clean_label][noise_dim]=100')
        group.add_argument('--train_gan', action='store_true',
                           help='whether train the GAN if it already exists, defaults to False')
        group.add_argument('--generator_iters', type=int,
                           help='epochs for training the generator, defaults to 1000')
        group.add_argument('--critic_iter', type=int,
                           help='critic iterations per generator training iteration '
                           '(default: 5)')
        return group

    def __init__(self, poison_generation_method: str = 'pgd',
                 pgd_alpha: float = 2 / 255, pgd_eps: float = 8 / 255, pgd_iter=7,
                 tau: float = 0.2, noise_dim: int = 100,
                 train_gan: bool = False, generator_iters: int = 1000, critic_iter: int = 5, **kwargs):
        super().__init__(**kwargs)
        self.param_list['clean_label'] = ['poison_generation_method', 'poison_num']
        self.poison_generation_method: str = poison_generation_method
        match poison_generation_method:
            case 'pgd':
                self.poison_num: int = int(len(self.dataset.get_dataset(
                    'train', class_list=[self.target_class])) * self.poison_percent)
            case 'gan':
                self.poison_num: int = int(len(self.dataset.get_dataset('train')) * self.poison_percent)

        match poison_generation_method:
            case 'pgd':
                self.param_list['pgd'] = ['pgd_alpha', 'pgd_eps', 'pgd_iter']
                if pgd_alpha is None:
                    pgd_alpha = 1.5 * pgd_eps / pgd_iter
                self.pgd_alpha: float = pgd_alpha
                self.pgd_eps: float = pgd_eps
                self.pgd_iter: int = pgd_iter
                self.pgd: PGD = PGD(pgd_alpha=pgd_alpha, pgd_eps=pgd_eps, iteration=pgd_iter,
                                    target_idx=0, output=self.output, dataset=self.dataset, model=self.model)
            case 'gan':
                self.param_list['gan'] = ['tau', 'noise_dim', 'train_gan', 'critic_iter', 'generator_iters']
                self.tau: float = tau
                self.noise_dim: int = noise_dim
                self.train_gan: bool = train_gan
                self.generator_iters = generator_iters
                self.critic_iter = critic_iter
                self.wgan = WGAN(noise_dim=self.noise_dim, dim=64, data_shape=self.dataset.data_shape,
                                 generator_iters=self.generator_iters, critic_iter=self.critic_iter)

    def attack(self, optimizer: torch.optim.Optimizer, lr_scheduler: torch.optim.lr_scheduler._LRScheduler, **kwargs):

        target_class_set = self.dataset.get_dataset('train', class_list=[self.target_class])
        target_imgs, _ = sample_batch(target_class_set, batch_size=self.poison_num)
        target_imgs = target_imgs.to(env['device'])

        full_set = self.dataset.get_dataset('train')
        poison_set: TensorListDataset = None    # TODO
        match self.poison_generation_method:
            case 'pgd':
                trigger_label = self.target_class * torch.ones(
                    len(target_imgs), dtype=torch.long, device=target_imgs.device)
                result = []
                for data in zip(target_imgs.chunk(self.dataset.batch_size),
                                trigger_label.chunk(self.dataset.batch_size)):
                    poison_img, _ = self.model.remove_misclassify(data)
                    poison_img, _ = self.pgd.optimize(poison_img)
                    poison_img = self.add_mark(poison_img).cpu()
                    result.append(poison_img)
                poison_imgs = torch.cat(result)

                poison_set = TensorListDataset(poison_imgs, [self.target_class] * len(poison_imgs))
                # poison_set = torch.utils.data.ConcatDataset([poison_set, target_original_dataset])

            case 'gan':
                other_classes = list(range(self.dataset.num_classes))
                other_classes.pop(self.target_class)
                x_list = []
                y_list = []
                for source_class in other_classes:
                    print('Process data of Source Class: ', source_class)
                    source_class_dataset = self.dataset.get_dataset(mode='train', class_list=[source_class])
                    sample_source_class_dataset, _ = self.dataset.split_dataset(
                        source_class_dataset, self.poison_num)
                    source_imgs = dataset_to_tensor(sample_source_class_dataset)[0].to(device=env['device'])

                    g_path = f'{self.folder_path}gan_dim{self.noise_dim}_class{source_class}_g.pth'
                    d_path = f'{self.folder_path}gan_dim{self.noise_dim}_class{source_class}_d.pth'
                    if os.path.exists(g_path) and os.path.exists(d_path) and not self.train_gan:
                        self.wgan.G.load_state_dict(torch.load(g_path, map_location=env['device']))
                        self.wgan.D.load_state_dict(torch.load(d_path, map_location=env['device']))
                        print(f'    load model from: \n        {g_path}\n        {d_path}', )
                    else:
                        self.train_gan = True
                        self.wgan.reset_parameters()
                        gan_dataset = torch.utils.data.ConcatDataset([source_class_dataset, target_class_set])
                        gan_dataloader = self.dataset.get_dataloader(
                            mode='train', dataset=gan_dataset, batch_size=self.dataset.batch_size, num_workers=0)
                        self.wgan.train(gan_dataloader)
                        torch.save(self.wgan.G.state_dict(), g_path)
                        torch.save(self.wgan.D.state_dict(), d_path)
                        print(f'GAN Model Saved at : \n{g_path}\n{d_path}')
                        continue

                    for source_chunk, target_chunk in zip(source_imgs.chunk(self.dataset.batch_size),
                                                          target_imgs.chunk(self.dataset.batch_size)):
                        source_encode = self.wgan.get_encode_value(source_chunk).detach()
                        target_encode = self.wgan.get_encode_value(target_chunk).detach()
                        # noise = torch.randn_like(source_encode)
                        # source_img = self.wgan.G(source_encode)
                        # target_img = self.wgan.G(target_encode)
                        # if not os.path.exists('./imgs'):
                        #     os.makedirs('./imgs')
                        # for i in range(len(source_img)):
                        #     F.to_pil_image(source_img[i]).save(f'./imgs/source_{i}.png')
                        # for i in range(len(target_img)):
                        #     F.to_pil_image(target_img[i]).save(f'./imgs/target_{i}.png')
                        # exit()
                        interpolation_encode = source_encode * self.tau + target_encode * (1 - self.tau)
                        poison_imgs = self.wgan.G(interpolation_encode).detach()
                        poison_imgs = self.add_mark(poison_imgs)

                        poison_imgs = poison_imgs.cpu()
                        x_list.append(poison_imgs)
                    y_list.extend([self.target_class] * len(source_imgs))
                assert not self.train_gan
                x_list = torch.cat(x_list)
                poison_set = TensorListDataset(x_list, y_list)
                # poison_set = torch.utils.data.ConcatDataset([poison_set, target_original_dataset])
        final_set = torch.utils.data.ConcatDataset([poison_set, full_set])
        # final_set = poison_set
        final_loader = self.dataset.get_dataloader(mode='train', dataset=final_set, num_workers=0)
        self.model._train(optimizer=optimizer, lr_scheduler=lr_scheduler, save_fn=self.save,
                          loader_train=final_loader, validate_fn=self.validate_fn, **kwargs)


class Generator(nn.Module):
    def __init__(self, noise_dim: int = 100, dim: int = 64, data_shape: list[int] = [3, 32, 32]):
        super().__init__()
        self.noise_dim: int = noise_dim
        self.dim: int = dim
        self.data_shape: list[int] = data_shape
        init_dim = dim * data_shape[1] * data_shape[2] // 16
        self.preprocess = nn.Linear(noise_dim, init_dim)
        self.preprocess_1 = nn.Sequential(
            nn.BatchNorm2d(init_dim),
            nn.ReLU(True),)
        self.block1 = nn.Sequential(
            nn.ConvTranspose2d(4 * dim, 2 * dim, 2, stride=2),
            nn.BatchNorm2d(2 * dim),
            nn.ReLU(True),)
        self.block2 = nn.Sequential(
            nn.ConvTranspose2d(2 * dim, dim, 2, stride=2),
            nn.BatchNorm2d(dim),
            nn.ReLU(True),)
        self.deconv_out = nn.ConvTranspose2d(dim, data_shape[0], 2, stride=2)
        self.tanh = nn.Tanh()

    def forward(self, x: torch.Tensor):
        # (N, noise_dim)
        x = self.preprocess(x)
        # (N, noise_dim)
        x = x.unsqueeze(-1).unsqueeze(-1)
        x = self.preprocess_1(x)
        x = x.view(len(x), 4 * self.dim, self.data_shape[1] // 8, self.data_shape[2] // 8)
        x = self.block1(x)
        x = self.block2(x)
        x = self.deconv_out(x)
        x = self.tanh(x)
        return x


class Discriminator(nn.Module):
    def __init__(self, dim: int = 64, data_shape: list = [3, 32, 32]):
        super(Discriminator, self).__init__()
        self.dim = dim
        self.main = nn.Sequential(
            nn.Conv2d(data_shape[0], dim, 3, 2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(dim, 2 * dim, 3, 2, padding=1),
            nn.BatchNorm2d(2 * dim),
            nn.LeakyReLU(),
            nn.Conv2d(2 * dim, 4 * dim, 3, 2, padding=1),
            nn.BatchNorm2d(4 * dim),
            nn.LeakyReLU(),
        )
        init_dim = dim * data_shape[1] * data_shape[2] // 16
        self.linear = nn.Linear(init_dim, 1)

    def forward(self, x: torch.Tensor):
        x = self.main(x)
        x = x.flatten(start_dim=1)
        x = self.linear(x)
        return x


class WGAN(object):
    def __init__(self, noise_dim: int, dim: int, data_shape: list[int] = [3, 32, 32],
                 generator_iters: int = 1000, critic_iter: int = 5):
        self.noise_dim = noise_dim
        self.G: Generator = Generator(noise_dim, dim, data_shape)
        self.D: Discriminator = Discriminator(dim, data_shape)
        if env['num_gpus']:
            self.G.cuda()
            self.D.cuda()
        # the parameter in the original paper
        self.d_optimizer = optim.RMSprop(self.D.parameters(), lr=5e-5)
        self.g_optimizer = optim.RMSprop(self.G.parameters(), lr=5e-5)
        self.generator_iters = generator_iters  # larger: 1000
        self.critic_iter = critic_iter
        self.mse_loss = torch.nn.MSELoss()

        self.gan_pgd = PGDoptimizer(pgd_eps=1.0, iteration=500, output=0)

    def reset_parameters(self):
        init_weights(self.G)
        init_weights(self.D)

    def train(self, train_dataloader):
        self.g_optimizer.zero_grad()
        self.d_optimizer.zero_grad()
        for g_iter in range(self.generator_iters):
            # Requires grad, Generator requires_grad = False
            for p in self.D.parameters():
                p.requires_grad = True
                p.data.clamp_(-0.01, 0.01)
            for p in self.G.parameters():
                p.requires_grad = False

            for d_iter in range(self.critic_iter):
                d_loss_fake = 0    # TODO
                d_loss_real = 0    # TODO
                for i, (data, label) in enumerate(train_dataloader):
                    data = torch.tensor(data)
                    train_data = data.to(env['device'])
                    d_loss_real = self.D(train_data).mean()

                    z = torch.randn(train_data.shape[0], self.noise_dim, device=train_data.device)
                    fake_images = self.G(z)
                    d_loss_fake = self.D(fake_images).mean()

                    d_loss = d_loss_fake - d_loss_real
                    d_loss.backward()
                    self.d_optimizer.step()
                    self.d_optimizer.zero_grad()
                print(f'    Discriminator: loss_fake: {d_loss_fake:.5f}, loss_real: {d_loss_real:.5f}')
            for p in self.D.parameters():
                p.requires_grad = False
            for p in self.G.parameters():
                p.requires_grad = True
            g_loss = 0    # TODO
            for i, (data, label) in enumerate(train_dataloader):
                data = torch.tensor(data)
                train_data = data.to(env['device'])
                z = torch.randn(train_data.shape[0], self.noise_dim, device=train_data.device)
                fake_images = self.G(z)
                g_loss = - self.D(fake_images).mean()
                g_loss.backward()
                self.g_optimizer.step()
                self.g_optimizer.zero_grad()
            print(f'Generator iteration: {g_iter:5d} / {self.generator_iters:5d}, g_loss: {g_loss:.5f}')

    def get_encode_value(self, imgs: torch.Tensor):
        """According to the image and Generator, utilize pgd optimization to get the d dimension encoding value.

        Args:
            imgs (torch.Tensor): the chosen image to get its encoding value, also considered as the output of Generator.
            noise_dim (int): the dimension of the input in the generator.

        Returns:
            torch.Tensor: the synthesized poisoned image.
        """

        def loss_func(X: torch.Tensor):
            loss = self.mse_loss(self.G(X), imgs)
            return loss
        x_1 = torch.randn(len(imgs), self.noise_dim, device=imgs.device)
        x_1, _ = self.gan_pgd.optimize(_input=x_1, loss_fn=loss_func)
        return x_1
