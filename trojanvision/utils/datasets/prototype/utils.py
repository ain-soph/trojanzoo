#!/usr/bin/env python3

import pathlib
from typing import Literal

from typing import TypedDict, NotRequired
from collections.abc import Callable


class _Resource(TypedDict):
    id: str
    file_name: str
    sha256: NotRequired[str]
    preprocess: NotRequired[Literal["decompress", "extract"] | Callable[[pathlib.Path], None]]
