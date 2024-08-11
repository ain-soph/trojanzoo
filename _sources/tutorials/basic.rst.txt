.. _tutorials.basic:

basic
======

You can view `example files at GitHub <https://github.com/ain-soph/trojanzoo/tree/main/examples>`_.

Following is `backdoor_defense.py <https://github.com/ain-soph/trojanzoo/blob/main/examples/backdoor_defense.py>`_
to show the basic workflow.

.. code-block:: python

    #!/usr/bin/env python3

    # CUDA_VISIBLE_DEVICES=0 python ./examples/backdoor_defense.py --color --verbose 1 --attack badnet --defense neural_cleanse --pretrained --validate_interval 1 --epochs 50 --lr 1e-2

    import trojanvision
    import argparse

    if __name__ == '__main__':
        parser = argparse.ArgumentParser()
        trojanvision.environ.add_argument(parser)
        trojanvision.datasets.add_argument(parser)
        trojanvision.models.add_argument(parser)
        trojanvision.trainer.add_argument(parser)
        trojanvision.marks.add_argument(parser)
        trojanvision.attacks.add_argument(parser)
        trojanvision.defenses.add_argument(parser)
        kwargs = parser.parse_args().__dict__

        env = trojanvision.environ.create(**kwargs)
        dataset = trojanvision.datasets.create(**kwargs)
        model = trojanvision.models.create(dataset=dataset, **kwargs)
        trainer = trojanvision.trainer.create(dataset=dataset, model=model, **kwargs)
        mark = trojanvision.marks.create(dataset=dataset, **kwargs)
        attack = trojanvision.attacks.create(dataset=dataset, model=model, mark=mark, **kwargs)
        defense = trojanvision.defenses.create(dataset=dataset, model=model, attack=attack, **kwargs)

        if env['verbose']:
            trojanvision.summary(env=env, dataset=dataset, model=model, mark=mark, trainer=trainer, attack=attack, defense=defense)
        defense.detect(**trainer)
