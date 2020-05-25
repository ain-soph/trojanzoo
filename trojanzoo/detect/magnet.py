# -*- coding: utf-8 -*-

from package.imports.universal import *
from package.utils.utils import *
from package.model.cnn import CNN, Conv2d_SAME, AverageMeter
import time





class _MagNet(nn.Module):
    """docstring for Model"""

    def __init__(self, layer=3, channel=3, **kwargs):
        super(_MagNet, self).__init__()

        # the location when loading pretrained weights using torch.load
        self.map_location = None if torch.cuda.is_available() else 'cpu'
        self.conv1 = Conv2d_SAME(channel, layer, kernel_size=(3, 3))
        self.bn1 = nn.BatchNorm2d(layer)
        self.relu1 = nn.Sigmoid()
        self.pool = nn.AvgPool2d(kernel_size=(2, 2))
        self.conv2 = Conv2d_SAME(layer, layer, kernel_size=(3, 3))
        self.bn2 = nn.BatchNorm2d(layer)
        self.relu2 = nn.Sigmoid()

        self.conv3 = Conv2d_SAME(layer, layer, kernel_size=(3, 3))
        self.bn3 = nn.BatchNorm2d(layer)
        self.relu3 = nn.Sigmoid()
        self.upsample = nn.Upsample(scale_factor=(2, 2))
        self.conv4 = Conv2d_SAME(layer, layer, kernel_size=(3, 3))
        self.bn4 = nn.BatchNorm2d(layer)
        self.relu4 = nn.Sigmoid()

        self.conv5 = Conv2d_SAME(layer, channel, kernel_size=(3, 3))
        self.bn5 = nn.BatchNorm2d(channel)
        self.sigmoid5 = nn.Sigmoid()

    # forward method
    # input: (batch_size, channels, height, width)
    # output: (batch_size, logits)

    def forward(self, x):
        # if x.shape is (channels, height, width)
        # (channels, height, width) ==> (batch_size: 1, channels, height, width)
        if len(x.shape) == 3:
            x.unsqueeze_(0)
        shape = x.shape
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.bn1(x)
        x = self.relu2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.upsample(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu4(x)

        x = self.conv5(x)
        x = self.bn5(x)
        x = self.sigmoid5(x)
        x = F.interpolate(x, size=shape[2:])
        return x


class MagNet(CNN):
    def __init__(self, name='magnet', dataset=None, model_class=_MagNet, structure=[3, "average", 3], v_noise=0.1, **kwargs):
        self.channel = 1 if dataset.name == 'mnist' else 3
        self.structure = structure
        self.v_noise = v_noise
        def func(data, v_noise=self.v_noise, mode='train'):
            _input = to_tensor(data[0])
            if mode == 'train':
                noise = to_tensor(torch.normal(mean=0.0, std=v_noise, size=_input.shape))
                noisy_input = to_valid_img(_input+noise)
                data[0] = noisy_input.detach()
                data[1] = _input.detach()
            else:
                data[0] = _input.detach()
                data[1] = _input.clone().detach()
            return data
        super(MagNet, self).__init__(
            name=name, dataset=dataset, model_class=model_class, channel=self.channel, get_data=func, **kwargs)

    # Define the optimizer
    # lr: (default: )
    # return: optimizer

    def define_optimizer(self, train_opt='full', optim_type='Adam', weight_decay=1e-9, **kwargs):
        return super(MagNet, self).define_optimizer(train_opt=train_opt, optim_type=optim_type, weight_decay=weight_decay, **kwargs)

    # define MSE loss function
    def define_criterion(self, *args, **kwargs):
        return nn.MSELoss()

    def load_pretrained_weights(self, **kwargs):
        return super(MagNet, self).load_pretrained_weights(**kwargs)

    def accuracy(self, _output, _label, topk=(1,)):
        res = []
        for k in topk:
            res.append(to_tensor([0.0]))
        return res
