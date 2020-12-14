trojanzoo.datasets
=================

All datasets are subclasses of :class:`trojanzoo.datasets.Dataset`
For example: ::

    imagenet_data = torjanzoo.datasets.ImageNet('path/to/imagenet_root/')
    data_loader = imagenet_data.get_dataloader(batch_size=4,
                                              shuffle=True)

The following datasets are available:

.. contents:: Datasets
    :local:

All the datasets have almost similar API.


.. currentmodule:: trojanzoo.datasets.image


CIFAR
~~~~~

.. autoclass:: CIFAR10
  :special-members:
  :members: __init__

.. note ::
    These require the `COCO API to be installed`_

.. _COCO API to be installed: https://github.com/pdollar/coco/tree/master/PythonAPI


Captions
^^^^^^^^

.. autoclass:: CIFAR10
  :special-members:
  :members: __init__
