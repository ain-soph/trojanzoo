#!/usr/bin/env python3

from trojanzoo.attacks import Attack

from .badnet import BadNet
from .trojannn import TrojanNN
from .latent_backdoor import LatentBackdoor
from .imc import IMC
from .refool import Refool
from .bypass_embed import BypassEmbed
from .trojannet import TrojanNet
from .clean_label import CleanLabel
from .hidden_trigger import HiddenTrigger
from .input_aware_dynamic import InputAwareDynamic

from .others import Unlearn

__all__ = ['BadNet', 'TrojanNN', 'LatentBackdoor',
           'IMC', 'Refool', 'BypassEmbed',
           'TrojanNet', 'CleanLabel', 'HiddenTrigger',
           'InputAwareDynamic',
           'Unlearn']

class_dict: dict[str, type[Attack]] = {
    'badnet': BadNet,
    'trojannn': TrojanNN,
    'latent_backdoor': LatentBackdoor,
    'imc': IMC,
    'refool': Refool,
    'bypass_embed': BypassEmbed,
    'trojannet': TrojanNet,
    'clean_label': CleanLabel,
    'hidden_trigger': HiddenTrigger,
    'input_aware_dynamic': InputAwareDynamic,

    'unlearn': Unlearn,
}
