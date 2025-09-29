from .im import *
from .standalone_image_utils import *
from .file_utils import *


def disable():
    from lovely_numpy import lovely, set_config

    set_config(repr=None)
