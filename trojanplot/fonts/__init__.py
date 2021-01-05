#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import matplotlib.font_manager

dirname = os.path.dirname(__file__)

for style in ['normal', 'italic', 'bold', 'bold_italic']:
    file_path = os.path.normpath(os.path.join(dirname, f'palatino_{style}.ttf'))
    matplotlib.font_manager.fontManager.addfont(file_path)
ttflist: list[matplotlib.font_manager.FontEntry] = matplotlib.font_manager.fontManager.ttflist
for i, font in enumerate(ttflist):
    if 'Palatino.ttc' in font.fname:
        del ttflist[i]
