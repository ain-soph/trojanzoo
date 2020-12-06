import os

for _file in os.listdir('./'):
    if 'percent' in _file:
        os.rename('./' + _file, './' + _file[:-16] + _file[-4:])
