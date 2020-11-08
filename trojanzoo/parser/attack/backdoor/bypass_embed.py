# -*- coding: utf-8 -*-

from .badnet import Parser_BadNet


class Parser_Bypass_Embed(Parser_BadNet):
    attack = 'bypass_embed'

    @classmethod
    def add_argument(cls, parser):
        super().add_argument(parser)
        parser.add_argument('--lambd', dest='lambd', type=int)
        parser.add_argument('--discrim_lr', dest='discrim_lr', type=float)
        parser.add_argument('--poison_num', dest='poison_num', type=int)
        