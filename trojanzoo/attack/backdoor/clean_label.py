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
# to optimize: gan train data; gan parameter; train set problem, source data not included;
# pgd.attack -> pgd; model.py 316 loss.backward(retain_graph=True);

class Clean_Label(BadNet):
    r"""
    Clean Label Backdoor Attack is described in detail in the paper `Clean Label Backdoor Attack`_ by Alexander Turner. 

    The main idea is to perturb the poisoned samples in order to render learning the salient characteristic of the input more difficult,causing the model rely more heavily on the backdoor pattern in order to successfully introduce backdoor. Utilize the adversarial examples and GAB generated data, the resulted poisoned inputs appear to be consistent with their label and thus seem benign even upon human inspection.

    The authors haven't posted `original source code`_.

    Args:
        preprocess_layer (str): the chosen layer used to generate adversarial example. Default: 'classifier'.
        poison_generation_method (str): the chosen method to generate poisoned sample. Default: 'PGD'.
        tau (float): the interpolation constant used to balance source imgs and target imgs. Default: 0.4.
        epsilon (float): the perturbation bound in input space. Default: 0.1.
        source_class (int): the class of sampled source data. Default: 0.
        target_class (int): the class of sampled target data. Default: 9.
        source_ratio (float): the ratio of source class sample used to generate poisoned sample, only for source class data,not whole training data. Default: 0.001.
        noise_dim (int): the dimension of the input in the generator. Default: 100.

    .. _Clean Label:
        https://people.csail.mit.edu/madry/lab/cleanlabel.pdf

    .. _related code:
        https://github.com/igul222/improved_wgan_training
        https://github.com/MadryLab/cifar10_challenge
        https://github.com/caogang/wgan-gp

    """
    name: str = 'clean_label'

    def __init__(self, preprocess_layer: str = 'classifier', poison_generation_method: str ='GAN', tau: float = 0.4, epsilon: float = 0.1, source_class: int = 0, target_class: int = 9, source_ratio: float = 0.001 , noise_dim: int = 100, **kwargs):
        super().__init__(**kwargs)
        
        self.preprocess_layer: str = preprocess_layer

        self.poison_generation_method: str = poison_generation_method

        self.tau: float = tau
        self.epsilon: float = epsilon

        self.source_class: int = source_class
        self.target_class: int = target_class

        self.source_ratio: float = source_ratio
        self.source_num: int = int(len(self.dataset.get_dataset('train', True, [source_class])) * source_ratio)

        data_shape = [self.dataset.n_channel]
        data_shape.extend(self.dataset.n_dim)
        self.data_shape: List[int] = data_shape
        
        self.source_dataloader = self.dataset.get_dataloader('train', full=True, classes=source_class, batch_size=self.source_num, shuffle=True, num_workers=0, drop_last=True)
        self.source_imgs, self.source_label = self.model.get_data(next(iter(self.source_dataloader)))

        self.target_dataloader = self.dataset.get_dataloader('train', full=True, classes=target_class, batch_size=self.source_num, shuffle=True, num_workers=0, drop_last=True)
        self.target_imgs, self.target_label = self.model.get_data(next(iter(self.target_dataloader)))

        # self.source_class_dataloader = self.dataset.get_dataloader('train', full=True, classes=source_class, batch_size=len(self.dataset.get_dataset('train', True, [source_class])), shuffle=True, num_workers=0, drop_last=True)
        # self.source_class_imgs, self.source_class_label = self.model.get_data(next(iter(self.source_class_dataloader)))

        # self.target_class_dataloader = self.dataset.get_dataloader('train', full=True, classes=target_class, batch_size=len(self.dataset.get_dataset('train', True, [source_class])), shuffle=True, num_workers=0, drop_last=True)
        # self.target_class_imgs, self.target_class_label = self.model.get_data(next(iter(self.target_class_dataloader)))

        self.noise_dim: int = noise_dim

        self.pgd: PGD = PGD(epsilon=epsilon, output=self.output)

        self.train_set = self.dataset.get_dataset('train', full=True, target_transform=torch.tensor)
        self.train_set = self.generate_train_set(self.train_set, self.target_imgs)

    def attack(self, optimizer: torch.optim.Optimizer, lr_scheduler: torch.optim.lr_scheduler._LRScheduler, iteration: int = None, **kwargs):
        if self.poison_generation_method == 'GAN':
            wgan = WGAN(self.noise_dim, self.source_num, self.data_shape)
            gan_data = torch.cat([self.source_imgs, self.target_imgs])

            # wgan = WGAN(self.noise_dim, len(self.dataset.get_dataset('train', True, [source_class])), self.data_shape)
            # gan_data = torch.cat([self.source_class_imgs, self.target_class_imgs])

            wgan.train(gan_data, self.noise_dim)
            source_encode, target_encode = wgan.get_encode_value(self.source_imgs, self.source_num, self.noise_dim), wgan.get_encode_value(self.target_imgs, self.source_num, self.noise_dim)
            interpolation_encode = source_encode * self.tau + target_encode * (1-self.tau)
            poison_imgs = wgan.G(interpolation_encode)
            poison_imgs = self.add_mark(poison_imgs)
            poison_set = torch.utils.data.TensorDataset(
                poison_imgs.to('cpu'), self.target_class * torch.ones(self.source_num, dtype=torch.long))

            final_set = torch.utils.data.ConcatDataset((poison_set, self.train_set))
            final_loader = self.dataset.get_dataloader(mode='train', dataset=final_set)
            self.model._train(optimizer=optimizer, lr_scheduler=lr_scheduler,
                            loader_train=final_loader, validate_func=self.validate_func, **kwargs)

        elif self.poison_generation_method == 'PGD':
            
            poison_imgs = self.generate_poisoned_data()
            poison_imgs = self.add_mark(poison_imgs)
            poison_set = torch.utils.data.TensorDataset(
                poison_imgs.to('cpu'), self.target_class * torch.ones(self.source_num, dtype=torch.long))
            
            final_set = torch.utils.data.ConcatDataset((poison_set, self.train_set))
            final_loader = self.dataset.get_dataloader(mode='train', dataset=final_set)
            self.model._train(optimizer=optimizer, lr_scheduler=lr_scheduler,
                            loader_train=final_loader, validate_func=self.validate_func, **kwargs)

        else:
            raise ValueError(self.poison_generation_method  + " poison generation method not supported.")
    
    # todo: Not Implemented
    def get_filename(self):
        return "filename"


    def loss(self, poison_imgs: torch.Tensor)-> torch.Tensor:
        """Compute the loss of generated poison image.

        Args:
            poison_imgs (torch.Tensor): the generated poison image.

        Returns:
            torch.Tensor: loss
        """
        loss = F.cross_entropy(self.model(poison_imgs), self.source_label)
        return -loss

    def validate_func(self, get_data: Callable[[torch.Tensor, torch.LongTensor], Tuple[torch.Tensor, torch.LongTensor]] = None, **kwargs) -> (float, float, float):
        self.model._validate(print_prefix='Validate Clean', **kwargs)
        self.model._validate(print_prefix='Validate Trigger Tgt', get_data=self.get_data, keep_org=False, **kwargs)
        self.model._validate(print_prefix='Validate Trigger Org',
                             get_data=self.get_data, keep_org=False, poison_label=False, **kwargs)
        return 0.0, 0.0, 0.0

    def generate_train_set(self, train_set, source_imgs):
        """Delete the sampled source image from the train set.

        Args:
            train_set (torch.utils.data.dataset): the initial train data set.
            source_imgs (torch.FloatTensor): the sampled target class images.

        Returns:
            torch.utils.data.dataset: train_set after deleting the sampled source class data
        """
        for i, data in enumerate(iter(train_set)):
            _input, _label = self.model.get_data(data)
            if i == 0:
                train_set_input = _input.view(1, _input.shape[0], _input.shape[1], _input.shape[2])
                train_set_label = torch.unsqueeze(_label,0)
            else:
                _input = _input.view(1, _input.shape[0], _input.shape[1], _input.shape[2])
                train_set_input = torch.cat((train_set_input, _input))
                train_set_label = torch.cat((train_set_label, torch.unsqueeze(_label, 0)))
        
        all_input = list(range(len(train_set)))
        idx = []
        for i in range(len(train_set)):
            if train_set_label[i].item() == self.source_class:
                for j in range(len(source_imgs)):
                    if torch.equal(source_imgs[j], train_set_input[i]):
                        idx.append(i)
        other_idx = list(set(all_input)-set(idx))
        train_set = torch.utils.data.Subset(train_set, other_idx)
        return  train_set

    def generate_poisoned_data(self) -> torch.Tensor:
        """Utilize pgd to get GAN-based perturbation synthesizing poisoned inputs.

        Returns:
            torch.Tensor: the poison images after pgd optimization.
        """
        noise = torch.zeros_like(self.target_imgs)
        def loss_func(poison_imgs):
            return self.loss(poison_imgs)
        poison_imgs, _ = self.pgd.optimize(_input=self.target_imgs, noise=noise,
                                               loss_fn=loss_func)
        return poison_imgs

class Generator(nn.Module):
    def __init__(self, noise_dim: int = 100, dim: int = 64, data_shape: list = [3, 32, 32]):
        super(Generator, self).__init__()
        self.dim =dim
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
        self.linear = nn.Linear(4*4*4*dim, 1)

    def forward(self, input):
        output = self.main(input)
        output = output.view(-1, 4*4*4*self.dim)
        output = self.linear(output)
        return output

class WGAN(object):
    def __init__(self, noise_dim, dim, data_shape):

        self.G = Generator(noise_dim, dim, data_shape)
        self.D = Discriminator(dim, data_shape)
        if env['num_gpus']:
            self.G.cuda()
            self.D.cuda()
        self.d_optimizer = optim.Adam(self.D.parameters(), lr=1e-4, betas=(0.5, 0.999))  # the parameter in the original paper
        self.g_optimizer = optim.Adam(self.G.parameters(), lr=1e-4, betas=(0.5, 0.999))
        self.generator_iters = 10  # larger: 1000
        self.critic_iter = 5

    def train(self, train_data,noise_dim):
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

                print(f'  Discriminator iteration: {d_iter}/{self.critic_iter}, loss_fake: {d_loss_fake}, loss_real: {d_loss_real}')

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

    def get_encode_value(self, imgs, source_num, noise_dim):
        """According to the image and Generator, utilize pgd optimization to get the d dimension encoding value.

        Args:
            imgs (torch.FloatTensor): the chosen image to get its encoding value, also considered as the output of Generator.
            source_num (int): the amount of chosen target class image.
            noise_dim (int): the dimension of the input in the generator.

        Returns:
            torch.FloatTensor: the synthesized poisoned image.
        """

        x_1 = torch.randn(source_num, noise_dim, device=env['device'])
        noise = torch.zeros_like(x_1, device=env['device'])
        self.gan_pgd : PGD = PGD(epsilon=1.0, iteration = 500, output= 0) 

        def loss_func(X: torch.Tensor):
            loss = torch.nn.MSELoss()(self.G(X), imgs)
            return loss
        
        cost = loss_func(x_1)
        x_1, _ = self.gan_pgd.optimize(_input=x_1, noise=noise,
                                               loss_fn=loss_func)    

        return x_1