# -*- coding: utf-8 -*-

import os
import matplotlib
import matplotlib.font_manager as font_manager

dirname = os.path.dirname(os.path.abspath(__file__))

palatino = font_manager.FontProperties(
    fname=dirname + '/palatino_normal.ttf')
palatino_bold = font_manager.FontProperties(
    fname=dirname + '/palatino_bold.ttf')
palatino_bold_italic = font_manager.FontProperties(
    fname=dirname + '/palatino_bold_italic.ttf')


font_files = font_manager.findSystemFonts(fontpaths=dirname)
font_list = font_manager.createFontList(font_files)
font_manager.fontManager.ttflist.extend(font_list)
