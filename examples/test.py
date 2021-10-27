# Used for python interactive window

import trojanvision

env = trojanvision.environ.create(verbose=1, color=True, tqdm=True)
dataset = trojanvision.datasets.create()
model = trojanvision.models.create(dataset=dataset)
trainer = trojanvision.trainer.create(dataset=dataset, model=model)
mark = trojanvision.marks.create(dataset=dataset)
attack = trojanvision.attacks.create('trojannn', dataset=dataset, model=model, mark=mark)
