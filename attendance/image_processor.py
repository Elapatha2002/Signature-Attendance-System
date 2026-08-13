# ==========================1. Attendance image processing module===========================================
"""Image-processing pipeline that turns a photographed signing sheet into
per-student attendance decisions.

The pipeline is deliberately rule based and every intermediate stage is written
to disk so that the result can be explained and audited:

    resize -> greyscale -> adaptive binarisation -> morphological line
    extraction -> table contour -> perspective correction -> grid detection ->
    per-cell ink analysis -> classification
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from .models import CourseInfo, ProcessingResult, RowDetection

ProgressCallback = Callable[[str], None]

# =======================2. Processing constants=================================

#: Fraction of a projected line profile that must be filled before the row or
#: column of pixels is accepted as a printed table separator.
SEPARATOR_FILL_RATIO = 0.5

#: Horizontal dilation, in pixels, used to rejoin a strike stroke that image
#: compression has broken into fragments.
STRIKE_BRIDGE = 5