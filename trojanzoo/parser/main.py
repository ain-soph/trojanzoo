# -*- coding: utf-8 -*-

from .parser import Parser
import torch

from trojanzoo.config import Config
env = Config.env


class Parser_Main(Parser):
    """ Main Parser for parameters in main scripts.

    :param name: ``'main'``.
    :type name: str
    """
    name = 'main'

    @staticmethod
    def add_argument(parser):
        parser.add_argument('--device', dest='device',
                            help='set to \'cpu\' to force cpu-only and \'gpu\', \'cuda\' for gpu-only, defaults to \'auto\'.')
        parser.add_argument('--benchmark', dest='benchmark', action='store_true',
                            help='use torch.backends.cudnn.benchmark to accelerate without deterministic, defaults to False.')
        parser.add_argument('--verbose', dest='verbose', action='store_true',
                            help='show arguments and module information, defaults to False.')

    @staticmethod
    def get_module(device: str = None, benchmark: bool = None, verbose: bool = None):
        """set default device and benchmark.

        :param device: set to ``\'cpu\'`` to force cpu-only and ``\'gpu\'``, ``\'cuda\'`` for gpu-only, defaults to ``\'auto\'``.
        :type device: str, optional
        :param benchmark: use ``torch.backends.cudnn.benchmark`` to accelerate without deterministic, defaults to ``False``
        :type benchmark: bool, optional
        :param verbose: show arguments and module information, defaults to ``False``
        :type verbose: bool, optional
        :raises Exception: GPU not enabled but ``device`` enforces gpu-only.
        """
        if verbose:
            env['verbose'] = verbose
        if device is None and 'device' in Config.config['env'].keys():
            device = Config.config['env']['device']
        env['device'] = 'cpu'
        env['num_gpus'] = 0
        if device in ['gpu', 'cuda', 'auto'] or 'cuda' in device:
            if torch.cuda.is_available():
                # torch.set_default_tensor_type(torch.cuda.FloatTensor)
                env['device'] = 'cuda'
                env['num_gpus'] = torch.cuda.device_count()
            elif device != 'auto':
                raise Exception('CUDA is not available on this device.')
        if benchmark:
            torch.backends.cudnn.benchmark = True
