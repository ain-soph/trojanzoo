trojanzoo.dataset
====================

All datasets are subclasses of :class:`trojanzoo.dataset.Dataset`
For example: ::

    imagenet_data = torjanzoo.dataset.ImageNet('path/to/imagenet_root/')
    data_loader = imagenet_data.get_dataloader(batch_size=4,
                                              shuffle=True)

The following datasets are available:

.. contents:: Dataset
    :local:

All the datasets have almost similar API.


.. currentmodule:: trojanzoo.dataset

Dataset
~~~~~~~

.. autoclass:: Dataset
  :special-members:
  :members: __init__
