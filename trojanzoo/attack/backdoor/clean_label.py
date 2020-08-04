from .badnet import BadNet

from trojanzoo.attack.adv import PGD

import torch.nn as nn
import torch.optim as optim

import numpy as np
import torch
from typing import List

from trojanzoo.utils import Config
env = Config.env
# to optimize: data augmentation;
# pgd.attack -> pgd; model.py 316 loss.backward(retain_graph=True);


class Clean_Label(BadNet):
    r"""
    Contributor: Xiangshan Gao

    Clean Label Backdoor Attack is described in detail in the paper `Clean Label Backdoor Attack`_ by Alexander Turner. 

    The main idea is to perturb the poisoned samples in order to render learning the salient characteristic of the input more difficult,causing the model rely more heavily on the backdoor pattern in order to successfully introduce backdoor. Utilize the adversarial examples and GAB generated data, the resulted poisoned inputs appear to be consistent with their label and thus seem benign even upon human inspection.

    The authors haven't posted `original source code`_.

    Args:
        preprocess_layer (str): the chosen layer used to generate adversarial example. Default: 'classifier'.
        poison_generation_method (str): the chosen method to generate poisoned sample. Default: 'pgd'.
        tau (float): the interpolation constant used to balance source imgs and target imgs. Default: 0.4.
        epsilon (float): the perturbation bound in input space. Default: 0.1.
        noise_dim (int): the dimension of the input in the generator. Default: 100.
        generator_iters (int): the epoch for training the generator. Default: 1000.
        critic_iter (int): the critic iterations per generator training iteration. Default: 5.


    .. _Clean Label:
        https://people.csail.mit.edu/madry/lab/cleanlabel.pdf

    .. _related code:
        https://github.com/igul222/improved_wgan_training
        https://github.com/MadryLab/cifar10_challenge
        https://github.com/caogang/wgan-gp

    """
    name: str = 'clean_label'

    def __init__(self, preprocess_layer: str = 'classifier', poison_generation_method: str = 'pgd',
                 pgd_alpha=30 / 255, pgd_epsilon: float = 300 / 255, pgd_iteration=20,
                 tau: float = 0.2, noise_dim: int = 100, generator_iters: int = 1000, critic_iter: int = 5, **kwargs):
        super().__init__(**kwargs)
        self.param_list['clean_label'] = ['preprocess_layer', 'poison_generation_method', 'poison_num']
        self.preprocess_layer: str = preprocess_layer
        self.poison_generation_method: str = poison_generation_method
        self.poison_num: int = int(len(self.dataset.get_dataset('train', True, [self.target_class])) * self.percent)

        data_shape = [self.dataset.n_channel]
        data_shape.extend(self.dataset.n_dim)
        self.data_shape: List[int] = data_shape
        if poison_generation_method == 'pgd':
            self.param_list['pgd'] = ['pgd_alpha', 'pgd_epsilon', 'pgd_iteration']
            self.pgd_alpha: float = pgd_alpha
            self.pgd_epsilon: float = pgd_epsilon
            self.pgd_iteration: int = pgd_iteration
            self.pgd: PGD = PGD(alpha=pgd_alpha, epsilon=pgd_epsilon, iteration=pgd_iteration,
                                target_idx=0, output=self.output)
        elif poison_generation_method == 'gan':
            self.param_list['gan'] = ['tau', 'noise_dim', 'critic_iter', 'generator_iters']
            self.tau: float = tau
            self.noise_dim: int = noise_dim
            self.generator_iters = generator_iters
            self.critic_iter = critic_iter
            self.wgan = WGAN(noise_dim=self.noise_dim, poison_num=self.poison_num, data_shape=self.data_shape,
                             generator_iters=self.generator_iters, critic_iter=self.critic_iter)

    def attack(self, optimizer: torch.optim.Optimizer, lr_scheduler: torch.optim.lr_scheduler._LRScheduler, **kwargs):
        other_classes = list(range(self.dataset.num_classes))
        other_classes.pop(self.target_class)
        target_class_dataset = self.dataset.get_dataset('train', full=True, classes=[self.target_class])
        poison_target_class_dataset, _ = self.dataset.split_set(
            target_class_dataset, self.poison_num)
        target_dataloader = self.dataset.get_dataloader(mode='train', dataset=poison_target_class_dataset,
                                                        batch_size=self.poison_num, num_workers=0)
        target_imgs, _ = self.model.get_data(next(iter(target_dataloader)))

        dataset_list = [target_class_dataset]
        for source_class in other_classes:
            source_class_dataset = self.dataset.get_dataset(mode='train', full=True, classes=[self.source_class])
            poison_source_class_dataset, other_source_class_dataset = self.dataset.split_set(
                source_class_dataset, self.poison_num)
            poison_source_class_dataloader = self.dataset.get_dataloader(mode='train', dataset=poison_source_class_dataset,
                                                                         batch_size=self.poison_num, num_workers=0)
            source_imgs, source_label = self.model.get_data(next(iter(poison_source_class_dataloader)))
            if self.poison_generation_method == 'gan':
                gan_data = torch.cat([source_imgs, target_imgs])
                wgan.train(gan_data)
                source_encode = wgan.get_encode_value(source_imgs)
                target_encode = wgan.get_encode_value(target_imgs)
                interpolation_encode = source_encode * self.tau + target_encode * (1 - self.tau)
                poison_imgs = wgan.G(interpolation_encode)
                poison_imgs = self.add_mark(poison_imgs)
            elif self.poison_generation_method == 'pgd':
                poison_imgs, _ = self.pgd.craft_example(_input=source_imgs)
                poison_imgs = self.add_mark(poison_imgs)
            else:
                raise ValueError(f'{self.poison_generation_method} poison generation method not supported.')
            poison_label = self.target_class * torch.ones(len(poison_imgs), dtype=torch.long)
            poison_set = torch.utils.data.TensorDataset(poison_imgs.to('cpu'), poison_label)
            dataset_list.append(poison_set)
            dataset_list.append(other_source_class_dataset)
        final_set = torch.utils.data.ConcatDataset(dataset_list)
        final_loader = self.dataset.get_dataloader(mode='train', dataset=final_set, num_workers=0)
        self.model._train(optimizer=optimizer, lr_scheduler=lr_scheduler,
                          loader_train=final_loader, validate_func=self.validate_func, **kwargs)

    # todo: Not Implemented
    def get_filename(self):
        return "filename"


class Generator(nn.Module):
    def __init__(self, noise_dim: int = 100, dim: int = 64, data_shape: List[int] = [3, 32, 32]):
        super().__init__()
        self.noise_dim: int = noise_dim
        self.dim: int = dim
        self.data_shape: List[int] = data_shape
        self.preprocess = nn.Linear(noise_dim, 4 * 4 * 4 * dim)
        self.preprocess_1 = nn.Sequential(
            nn.BatchNorm2d(4 * 4 * 4 * dim),
            nn.ReLU(True),)
        self.block1 = nn.Sequential(
            nn.ConvTranspose2d(4 * dim, 2 * dim, 2, stride=2),
            nn.BatchNorm2d(2 * dim),
            nn.ReLU(True),)
        self.block2 = nn.Sequential(
            nn.ConvTranspose2d(2 * dim, dim, 2, stride=2),
            nn.BatchNorm2d(dim),
            nn.ReLU(True),)
        self.deconv_out = nn.ConvTranspose2d(dim, 3, 2, stride=2)
        self.tanh = nn.Tanh()

    def forward(self, x: torch.Tensor):
        x = self.preprocess(x)
        x = x.unsqueeze(-1).unsqueeze(-1)
        x = self.preprocess_1(x)
        x = x.view(-1, 4 * self.dim, 4, 4)
        x = self.block1(x)
        x = self.block2(x)
        x = self.deconv_out(x)
        x = self.tanh(x)
        return x.view(-1, self.data_shape[0], self.data_shape[1], self.data_shape[2])


class Discriminator(nn.Module):
    def __init__(self, dim: int = 64, data_shape: list = [3, 32, 32]):
        super(Discriminator, self).__init__()
        self.dim = dim
        self.main = nn.Sequential(
            nn.Conv2d(data_shape[0], dim, 3, 2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(dim, 2 * dim, 3, 2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(2 * dim, 4 * dim, 3, 2, padding=1),
            nn.LeakyReLU(),
        )
        self.linear = nn.Linear(4 * 4 * 4 * dim, 1)

    def forward(self, x: torch.Tensor):
        x = self.main(x)
        x = x.view(-1, 4 * 4 * 4 * self.dim)
        x = self.linear(x)
        return x


class WGAN(object):
    def __init__(self, noise_dim: int, dim: int, data_shape: List[int] = [3, 32, 32],
                 generator_iters: int = 1000, critic_iter: int = 5):
        self.G = Generator(noise_dim, dim, data_shape)
        self.D = Discriminator(dim, data_shape)
        if env['num_gpus']:
            self.G.cuda()
            self.D.cuda()
        self.d_optimizer = optim.Adam(self.D.parameters(), lr=1e-4, betas=(0.5, 0.999)
                                      )  # the parameter in the original paper
        self.g_optimizer = optim.Adam(self.G.parameters(), lr=1e-4, betas=(0.5, 0.999))
        self.generator_iters = generator_iters  # larger: 1000
        self.critic_iter = critic_iter

    def train(self, train_data):
        # Now batches are callable self.data.next()
        self.data = train_data
        one = torch.tensor(1, dtype=torch.float)
        mone = torch.tensor(-1, dtype=torch.float)
        if env['num_gpus']:
            one = one.cuda()
            mone = mone.cuda()
        for g_iter in range(self.generator_iters):
            # Requires grad, Generator requires_grad = False
            for p in self.D.parameters():
                p.requires_grad = True

            d_loss_real = 0
            d_loss_fake = 0
            Wasserstein_D = 0
            # Train Dicriminator forward-loss-backward-update self.critic_iter times while 1 Generator forward-loss-backward-update

            for d_iter in range(self.critic_iter):
                self.D.zero_grad()
                images = self.data
                # Check for batch to have full batch_size
                z = torch.rand(images.shape[0], noise_dim, 1, 1, device=env['device'])

                # Train discriminator
                # WGAN - Training discriminator more iterations than generator
                # Train with real images
                d_loss_real = self.D(images)
                d_loss_real = d_loss_real.mean()
                d_loss_real.backward(mone)
                # Train with fake images

                z = torch.randn(images.shape[0], noise_dim, device=env['device'])
                print(z.shape)
                fake_images = self.G(z)
                d_loss_fake = self.D(fake_images)
                d_loss_fake = d_loss_fake.mean()
                d_loss_fake.backward(one)

                d_loss = d_loss_fake - d_loss_real
                Wasserstein_D = d_loss_real - d_loss_fake
                self.d_optimizer.step()

                print(
                    f'  Discriminator iteration: {d_iter}/{self.critic_iter}, loss_fake: {d_loss_fake}, loss_real: {d_loss_real}')

            # Generator update
            for p in self.D.parameters():
                p.requires_grad = False  # to avoid computation

            self.G.zero_grad()
            # train generator
            # compute loss with fake images
            z = torch.randn(images.shape[0], noise_dim, device=env['device'])
            fake_images = self.G(z)
            g_loss = self.D(fake_images)
            g_loss = g_loss.mean()
            g_loss.backward(mone)
            g_cost = -g_loss
            self.g_optimizer.step()
            print(f'Generator iteration: {g_iter}/{self.generator_iters}, g_loss: {g_loss}')

    def get_encode_value(self, imgs, poison_num, noise_dim):
        """According to the image and Generator, utilize pgd optimization to get the d dimension encoding value.

        Args:
            imgs (torch.FloatTensor): the chosen image to get its encoding value, also considered as the output of Generator.
            poison_num (int): the amount of chosen target class image.
            noise_dim (int): the dimension of the input in the generator.

        Returns:
            torch.FloatTensor: the synthesized poisoned image.
        """

        x_1 = torch.randn(poison_num, noise_dim, device=env['device'])
        noise = torch.zeros_like(x_1, device=env['device'])
        self.gan_pgd: PGD = PGD(epsilon=1.0, iteration=500, output=0)

        def loss_func(X: torch.Tensor):
            loss = torch.nn.MSELoss()(self.G(X), imgs)
            return loss

        cost = loss_func(x_1)
        x_1, _ = self.gan_pgd.optimize(_input=x_1, noise=noise,
                                       loss_fn=loss_func)

        return x_1
