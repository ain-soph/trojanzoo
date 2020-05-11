import os
from package.parse.model import Parser_Model
from package.detect.abs import ABS
import argparse
parser = Parser_Model()
dataset = parser.module.dataset
model = parser.module.model
detector = ABS(model)


parser = argparse.ArgumentParser()
parser.add_argument('--max_troj_size',
                    dest='max_troj_size', default=64, type=int)
args, unknown = parser.parse_known_args()


detector.max_troj_size = args.max_troj_size

detector.trojan_prob(model.folder_path + model.name + '.pth')
