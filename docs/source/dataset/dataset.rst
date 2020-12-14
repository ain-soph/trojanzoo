trojanzoo.datasets
====================

All datasets are subclasses of :class:`trojanzoo.datasets.Dataset`
For example: ::

    imagenet_data = torjanzoo.dataset.ImageNet('path/to/imagenet_root/')
    data_loader = imagenet_data.get_dataloader(batch_size=4,
                                              shuffle=True)

The following datasets are available:

.. contents:: Dataset
    :local:

All the datasets have almost similar API.


.. currentmodule:: trojanzoo.datasets

Dataset
~~~~~~~

.. autoclass:: Dataset
  :special-members:
  :members: __init__
