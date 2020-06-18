import os
import kaggle
from easydict import EasyDict as edict

_C = edict()
config = _C

_C.BASEPATH = os.path.abspath(os.pardir)

# Data folders
_C.DATA = edict()
_C.DATA.BASE = os.path.join(_C.BASEPATH, "data")