# -*- coding: utf-8 -*-

from .badnet import Parser_BadNet


class Parser_TrojanNet(Parser_BadNet):
    attack = 'trojannet'
