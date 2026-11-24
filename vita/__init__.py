"""Independent VITA reference; smoke tests are not benchmark reproduction."""

from .config import VITAConfig
from .model import VITA

__all__ = ["VITA", "VITAConfig"]
