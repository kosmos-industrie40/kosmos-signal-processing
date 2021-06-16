"""
This module provides wrapping logic for interpolation methods
"""

from kosmos_signal_processing.interpolate.interpol import (
    interpol_method_selection,
    selection_mask,
    do_interpolate,
)
from kosmos_signal_processing.interpolate.trig import (
    eval_sin_cos_representation,
    get_sin_cos_representation,
)
