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

# =========================3. Table detection error===================
class TableDetectionError(RuntimeError):
    """Raised when the student table cannot be located in a photograph."""

#=======================4. Processor class and initialization=====================
class AttendanceImageProcessor:
    """Detect the student table and classify the signature cell of each row."""

    def __init__(
        self,
        course: CourseInfo,
        target_width: int = 1400,
        table_width: int = 1200,
        min_table_height: int = 300,
        max_table_height: int = 640,
    ) -> None:
        self.course = course
        self.target_width = target_width
        self.table_width = table_width
        self.min_table_height = min_table_height
        self.max_table_height = max_table_height