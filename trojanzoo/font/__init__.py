# -*- coding: utf-8 -*-

import os
import matplotlib.font_manager

dirname = os.path.dirname(__file__)

palatino = matplotlib.font_manager.FontProperties(
    fname=dirname+'/palatino_normal.ttf')
palatino_bold = matplotlib.font_manager.FontProperties(
    fname=dirname+'/palatino_bold.ttf')
