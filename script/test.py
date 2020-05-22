
import argparse
from package.parse.model import Parser_Model

parser_model = Parser_Model()
dataset = parser_model.module.dataset
model = parser_model.module.model
_, org_acc, _ = model._validate(full=True)

