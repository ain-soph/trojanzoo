#!/usr/bin/env python3

r"""It is equivalent to :ref:`trojanzoo.environ <trojanzoo.environ>`.

Note:
    The only difference is that it uses ``trojanvision.configs.config``
    as the default parameter passed to :func:`trojanzoo.environ.create()`.
"""

from trojanvision.configs import config
from trojanzoo.environ import add_argument, env  # noqa: F401
import trojanzoo.environ

from trojanvision.configs import Config


def create(config: Config = config, **kwargs) -> trojanzoo.environ.Env:
    return trojanzoo.environ.create(config=config, **kwargs)


create()
