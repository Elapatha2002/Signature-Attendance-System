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

 # ================= Main Utility functions =========================


    #===================5. Point ordering========================
    @staticmethod
    def _order_points(points: np.ndarray) -> np.ndarray:
        """Return the four corners ordered top-left, top-right, bottom-right, bottom-left."""
        points = np.asarray(points, dtype=np.float32)
        sums = points.sum(axis=1)
        differences = np.diff(points, axis=1).ravel()
        return np.array(
            [
                points[np.argmin(sums)],
                points[np.argmin(differences)],
                points[np.argmax(sums)],
                points[np.argmax(differences)],
            ],
            dtype=np.float32,
        )

    #======================6. Image saving========================
        @staticmethod
        def _save(path: Path, image: np.ndarray) -> None:
            path.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(path), image):
                raise OSError(f"Could not write image: {path}")

    #==================== 7: Contrast normalization=====================
        @staticmethod
        def normalize_contrast(greyscale: np.ndarray) -> np.ndarray:
            """Equalise local contrast so that exposure no longer changes the result.
    
            Contrast Limited Adaptive Histogram Equalisation redistributes intensity
            inside small tiles. Without it the fixed offset used by the adaptive
            threshold below silently stops finding the printed rules on an
            under-exposed photograph, and the table can no longer be detected.
            """
            clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
            return clahe.apply(greyscale)

     #==================== 8: Adaptive binarization=====================
        @classmethod
        def _binarize(cls, image: np.ndarray) -> np.ndarray:
            """Adaptive binarisation that keeps dark ink and printed lines as white."""
            greyscale = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            return cv2.adaptiveThreshold(
                cv2.bitwise_not(cls.normalize_contrast(greyscale)),
                255,
                cv2.ADAPTIVE_THRESH_MEAN_C,
                cv2.THRESH_BINARY,
                31,
                -10,
            )

     #==================== 9: Skew estimation=====================
        @staticmethod
        def estimate_skew(binary: np.ndarray, maximum_angle: float = 20.0) -> float:
            """Estimate the page rotation in degrees from the printed table rules.
    
            Morphological line extraction relies on strictly horizontal and vertical
            structuring elements, so a tilted photograph breaks the grid into short
            fragments and the table can no longer be found. A probabilistic Hough
            transform gives the dominant direction of the long printed rules and the
            median of those angles is used to level the page first.
            """
            height, width = binary.shape
            segments = cv2.HoughLinesP(
                binary,
                rho=1,
                theta=np.pi / 720,
                threshold=120,
                minLineLength=int(width * 0.25),
                maxLineGap=12,
            )
            if segments is None:
                return 0.0
    
            angles: list[float] = []
            for x1, y1, x2, y2 in segments.reshape(-1, 4):
                angle = float(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
                if angle > 90:
                    angle -= 180
                elif angle < -90:
                    angle += 180
                if abs(angle) <= maximum_angle:
                    angles.append(angle)
    
            if len(angles) < 3:
                return 0.0
            return float(np.median(angles))

    #==================== 10: Image rotation=====================
        @staticmethod
        def rotate_upright(image: np.ndarray, angle: float) -> np.ndarray:
            """Rotate ``image`` by ``-angle`` degrees, growing the canvas to fit."""
            height, width = image.shape[:2]
            centre = (width / 2.0, height / 2.0)
            matrix = cv2.getRotationMatrix2D(centre, angle, 1.0)
            cosine, sine = abs(matrix[0, 0]), abs(matrix[0, 1])
            new_width = int(height * sine + width * cosine)
            new_height = int(height * cosine + width * sine)
            matrix[0, 2] += new_width / 2.0 - centre[0]
            matrix[1, 2] += new_height / 2.0 - centre[1]
            # The corners exposed by the rotation are filled by replicating the edge
            # rather than with a constant colour. A constant fill introduces a hard
            # page-sized rectangle that the line morphology then mistakes for a table
            # border and merges with the real grid.
            return cv2.warpAffine(
                image,
                matrix,
                (new_width, new_height),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REPLICATE,
            )

    #==================== 11: Morphological kernels=====================
    
        def _find_student_table(
            self, image: np.ndarray, binary: np.ndarray
        ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            """Locate the student table and remove rotation and perspective distortion."""
            height, width = binary.shape
            horizontal_kernel = cv2.getStructuringElement(
                cv2.MORPH_RECT, (max(25, width // 35), 1)
            )
            vertical_kernel = cv2.getStructuringElement(
                cv2.MORPH_RECT, (1, max(25, height // 60))
            )
    #==================== 12: Extract horizontal and vertical table lines=====================
            horizontal = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel)
            vertical = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel)
            grid = cv2.bitwise_or(horizontal, vertical)

    #==================== 13: Improve table grid connectivity=====================
            grid = cv2.morphologyEx(
                grid,
                cv2.MORPH_CLOSE,
                cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)),
                iterations=2,
            )
    #==================== 14: Detect student table contours=====================
            contours, _ = cv2.findContours(grid, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            row_count = len(self.course.students)
            candidates: list[np.ndarray] = []

     #==================== 15: filter student table candidates=====================
            for contour in contours:
                _, _, box_width, box_height = cv2.boundingRect(contour)
                aspect_ratio = box_width / max(box_height, 1)
                # A signing-sheet table is a wide, short block: wide enough to be the
                # table rather than a word, short enough not to be the whole page,
                # and tall enough to hold one row per student.
                wide_enough = box_width > width * 0.30
                tall_enough = box_height > 20 * row_count
                not_the_whole_page = box_height < height * 0.75
                plausible_shape = 1.5 < aspect_ratio < 8.0
                if wide_enough and tall_enough and not_the_whole_page and plausible_shape:
                    candidates.append(contour)

      #==================== 16: Handle missing student table=====================
            if not candidates:
                raise TableDetectionError(
                    "The student table could not be detected. Use a clear photograph "
                    "that shows the complete table including its outer border."
                )
    
    