import sys

from .data import process as data_process
from .models import modules
from .models.core import (MCANModel, BinaryClassifier)

sys.modules['mcan.modules'] = modules

__all__ = [
    'modules'
]

_check_nan = True


def disable_nan_checks():
    _check_nan = False


def enable_nan_checks():
    _check_nan = True
