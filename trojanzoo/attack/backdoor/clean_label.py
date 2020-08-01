from .badnet import BadNet

from trojanzoo.optim import PGD

import torch.optim as optim
import numpy as np
import torch
from typing import Tuple, Callable
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
from trojanzoo.utils import Config
env = Config.env
# to optimize: data augmentation;
# pgd.attack -> pgd; model.py 316 loss.backward(retain_graph=True);


class Clean_Label(BadNet):
    r"""
    Contribute by Xiangshan Gao

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
                 tau: float = 0.2, epsilon: float = 0.1,
                 noise_dim: int = 100, generator_iters: int = 1000, critic_iter: int = 5, **kwargs):
        super().__init__(**kwargs)

        self.preprocess_layer: str = preprocess_layer
        self.poison_generation_method: str = poison_generation_method

        self.tau: float = tau
        self.epsilon: float = epsilon

        self.poison_num: int = int(len(self.dataset.get_dataset('train', True, [self.target_class])) * self.percent)

        data_shape = [self.dataset.n_channel]
        data_shape.extend(self.dataset.n_dim)
        self.data_shape: List[int] = data_shape

        self.noise_dim: int = noise_dim
        self.pgd: PGD = PGD(epsilon=epsilon, output=self.output)

        self.generator_iters = generator_iters
        self.critic_iter = critic_iter

    def attack(self, optimizer: torch.optim.Optimizer, lr_scheduler: torch.optim.lr_scheduler._LRScheduler, iteration: int = None, **kwargs):
        other_classes = list(range(self.dataset.num_classes))
        other_classes.pop(self.target_class)
        other_class_dataset = self.dataset.get_dataset(mode='train', full=True, classes=other_classes)
        if self.poison_generation_method == 'gan':
            target_class_dataset = self.dataset.get_dataset('train', full=True, classes=[self.target_class])
            poison_target_class_dataset, other_target_class_dataset = self.dataset.split_set(
                target_class_dataset, self.poison_num)
            target_dataloader = self.dataset.get_dataloader(mode='train', dataset=poison_target_class_dataset,
                                                            batch_size=self.poison_num, num_workers=0)
            target_imgs, _ = self.model.get_data(next(iter(target_dataloader)))
            for source_class in other_classes:
                source_class_dataset = self.dataset.get_dataset(mode='train', full=True, classes=[self.source_class])
                source_dataloader = self.dataset.get_dataloader(mode='train', dataset=source_class_dataset,
                                                                batch_size=self.poison_num, num_workers=0)
                source_imgs, source_label = self.model.get_data(next(iter(self.source_dataloader)))
                gan_data = torch.cat([source_imgs, target_imgs])

                wgan = WGAN(self.noise_dim, self.poison_num, self.data_shape,
                            self.generator_iters, self.critic_iter)
                wgan.train(gan_data, self.noise_dim)
                source_encode = wgan.get_encode_value(source_imgs, self.poison_num, self.noise_dim)
                target_encode = wgan.get_encode_value(target_imgs, self.poison_num, self.noise_dim)
                interpolation_encode = source_encode * self.tau + target_encode * (1 - self.tau)
                poison_imgs = wgan.G(interpolation_encode)
                poison_imgs = self.add_mark(poison_imgs)
                poison_set = torch.utils.data.TensorDataset(
                    poison_imgs.to('cpu'), self.target_class * torch.ones(self.poison_num, dtype=torch.long))

                for i, data in enumerate(iter(poison_set)):
                    _input, _label = self.model.get_data(data)

                    poison_input = _input.view(1, _input.shape[0], _input.shape[1], _input.shape[2])
                    if i == 0:
                        poison_input_all = poison_input
                        label_all = torch.unsqueeze(_label, 0)
                    else:
                        poison_input_all = torch.cat((poison_input_all, poison_input))
                        label_all = torch.cat((label_all, torch.unsqueeze(_label, 0)))
                label_all = torch.squeeze(label_all, 0)

                poison_set = torch.utils.data.dataset.TensorDataset(poison_input_all, label_all)

                for i, data in enumerate(iter(self.other_target_class_dataset)):
                    _input, _label = self.model.get_data(data)

                    poison_input = _input.view(1, _input.shape[0], _input.shape[1], _input.shape[2])
                    if i == 0:
                        poison_input_all = poison_input
                        label_all = torch.unsqueeze(_label, 0)
                    else:
                        poison_input_all = torch.cat((poison_input_all, poison_input))
                        label_all = torch.cat((label_all, torch.unsqueeze(_label, 0)))
                label_all = torch.squeeze(label_all, 0)

                self.other_target_class_dataset = torch.utils.data.dataset.TensorDataset(
                    poison_input_all, label_all)

                final_target_class_set = torch.utils.data.ConcatDataset(
                    [poison_set, self.other_target_class_dataset])

                for i, data in enumerate(iter(self.other_class_dataset)):
                    _input, _label = self.model.get_data(data)

                    poison_input = _input.view(1, _input.shape[0], _input.shape[1], _input.shape[2])
                    if i == 0:
                        poison_input_all = poison_input
                        label_all = torch.unsqueeze(_label, 0)
                    else:
                        poison_input_all = torch.cat((poison_input_all, poison_input))
                        label_all = torch.cat((label_all, torch.unsqueeze(_label, 0)))
                label_all = torch.squeeze(label_all, 0)

                self.other_class_dataset = torch.utils.data.dataset.TensorDataset(poison_input_all, label_all)

                final_set = torch.utils.data.ConcatDataset([final_target_class_set, self.other_class_dataset])
                final_loader = self.dataset.get_dataloader(
                    mode=None, dataset=final_set, num_workers=0, pin_memory=False)

                self.model._train(optimizer=optimizer, lr_scheduler=lr_scheduler,
                                  loader_train=final_loader, validate_func=self.validate_func, **kwargs)

        elif self.poison_generation_method == 'pgd':

            poison_imgs = self.generate_poisoned_data()
            poison_imgs = self.add_mark(poison_imgs)
            poison_set = torch.utils.data.TensorDataset(
                poison_imgs.to('cpu'), self.target_class * torch.ones(self.poison_num, dtype=torch.long))

            final_target_class_set = torch.utils.data.ConcatDataset([poison_set, self.other_target_class_dataset])
            final_set = torch.utils.data.ConcatDataset([final_target_class_set, self.other_class_dataset])
            final_loader = self.dataset.get_dataloader(mode=None, dataset=final_set)
            self.model._train(optimizer=optimizer, lr_scheduler=lr_scheduler,
                              loader_train=final_loader, validate_func=self.validate_func, **kwargs)

        else:
            raise ValueError(f'{self.poison_generation_method} poison generation method not supported.')

    # todo: Not Implemented
    def get_filename(self):
        return "filename"

    def loss(self, poison_imgs: torch.Tensor) -> torch.Tensor:
        """Compute the loss of generated poison image.

        Args:
            poison_imgs (torch.Tensor): the generated poison image.

        Returns:
            torch.Tensor: loss
        """
        loss = F.cross_entropy(self.model(poison_imgs), self.source_label)
        return -loss

    def generate_poisoned_data(self) -> torch.Tensor:
        """Utilize pgd to get GAN-based perturbation synthesizing poisoned inputs.

        Returns:
            torch.Tensor: the poison images after pgd optimization.
        """
        noise = torch.zeros_like(self.target_imgs)

        poison_imgs, _ = self.pgd.optimize(_input=self.target_imgs, noise=noise,
                                           loss_fn=self.loss)
        return poison_imgs


class Generator(nn.Module):
    def __init__(self, noise_dim: int = 100, dim: int = 64, data_shape: list = [3, 32, 32]):
        super(Generator, self).__init__()
        self.dim = dim
        self.data_shape = data_shape
        preprocess = nn.Sequential(
            nn.Linear(noise_dim, 4 * 4 * 4 * dim)
        )
        preprocess_1 = nn.Sequential(
            nn.BatchNorm2d(4 * 4 * 4 * dim),
            nn.ReLU(True),
        )
        block1 = nn.Sequential(
            nn.ConvTranspose2d(4 * dim, 2 * dim, 2, stride=2),
            nn.BatchNorm2d(2 * dim),
            nn.ReLU(True),
        )
        block2 = nn.Sequential(
            nn.ConvTranspose2d(2 * dim, dim, 2, stride=2),
            nn.BatchNorm2d(dim),
            nn.ReLU(True),
        )
        deconv_out = nn.ConvTranspose2d(dim, 3, 2, stride=2)

        self.preprocess = preprocess
        self.preprocess_1 = preprocess_1
        self.block1 = block1
        self.block2 = block2
        self.deconv_out = deconv_out
        self.tanh = nn.Tanh()

    def forward(self, input):
        output = self.preprocess(input)
        output = output.view(output.shape[0], output.shape[1], 1, 1)
        output = self.preprocess_1(output)
        output = output.view(-1, 4 * self.dim, 4, 4)
        output = self.block1(output)
        output = self.block2(output)
        output = self.deconv_out(output)
        output = self.tanh(output)
        return output.view(-1, self.data_shape[0], self.data_shape[1], self.data_shape[2])


class Discriminator(nn.Module):
    def __init__(self, dim: int = 64, data_shape: list = [3, 32, 32]):
        super(Discriminator, self).__init__()
        self.dim = dim
        main = nn.Sequential(
            nn.Conv2d(data_shape[0], dim, 3, 2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(dim, 2 * dim, 3, 2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(2 * dim, 4 * dim, 3, 2, padding=1),
            nn.LeakyReLU(),
        )
        self.main = main
        self.linear = nn.Linear(4 * 4 * 4 * dim, 1)

    def forward(self, input):
        output = self.main(input)
        output = output.view(-1, 4 * 4 * 4 * self.dim)
        output = self.linear(output)
        return output


class WGAN(object):
    def __init__(self, noise_dim: int, dim: int, data_shape, generator_iters, critic_iter):
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

    def train(self, train_data, noise_dim):
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
