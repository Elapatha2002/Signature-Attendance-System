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
    #==================== 17: Select student table contour=====================
            contour = max(candidates, key=cv2.contourArea)
            perimeter = cv2.arcLength(contour, True)
            polygon = cv2.approxPolyDP(contour, 0.02 * perimeter, True).reshape(-1, 2)
    
    #==================== 18: Detect table quadrilateral corners=====================
            if len(polygon) != 4:
                polygon = cv2.boxPoints(cv2.minAreaRect(contour))
            quadrilateral = self._order_points(polygon)

     #==================== 19: Calculate perspective correction dimensions=====================
            table_height = self._destination_height(quadrilateral)
            destination = np.array(
                [
                    [0, 0],
                    [self.table_width - 1, 0],
                    [self.table_width - 1, table_height - 1],
                    [0, table_height - 1],
                ],
                dtype=np.float32,
            )
    #==================== 20: Student table perspective correction=====================
            matrix = cv2.getPerspectiveTransform(quadrilateral, destination)
            warped = cv2.warpPerspective(image, matrix, (self.table_width, table_height))
            return warped, horizontal, vertical, quadrilateral

    #==================== 21: Calculate table physical aspect ratio=====================
        def _destination_height(self, quadrilateral: np.ndarray) -> int:
            """Choose a warp height that preserves the physical aspect ratio of the table."""
            top_left, top_right, bottom_right, bottom_left = quadrilateral
            source_width = max(
                np.linalg.norm(top_right - top_left),
                np.linalg.norm(bottom_right - bottom_left),
            )
            source_height = max(
                np.linalg.norm(bottom_left - top_left),
                np.linalg.norm(bottom_right - top_right),
            )

    #==================== 22: Constrain corrected table height=====================
            ratio = float(source_height) / max(float(source_width), 1.0)
            return int(
                np.clip(
                    round(self.table_width * ratio),
                    self.min_table_height,
                    self.max_table_height,
                )
            )
        # --------------------------------------------------------- grid detection

    @staticmethod
    def _separator_positions(line_mask: np.ndarray, axis: int) -> list[int]:
        """Return the centre position of every printed separator in ``line_mask``.

        ``axis=0`` projects onto the rows and therefore returns the y positions of
        horizontal separators; ``axis=1`` returns the x positions of vertical ones.
        """
        profile = line_mask.sum(axis=1 if axis == 0 else 0) / 255.0
        span = line_mask.shape[1] if axis == 0 else line_mask.shape[0]
        filled = profile > SEPARATOR_FILL_RATIO * span

        positions: list[int] = []
        current_run: list[int] = []
        for index, is_filled in enumerate(filled):
            if is_filled:
                current_run.append(index)
            elif current_run:
                positions.append(int(round(float(np.mean(current_run)))))
                current_run = []
        if current_run:
            positions.append(int(round(float(np.mean(current_run)))))
        return positions

    def _detect_grid(
        self, warped_binary: np.ndarray
    ) -> tuple[list[int], list[int], np.ndarray, np.ndarray]:
        """Detect the horizontal and vertical separators of the corrected table."""
        height, width = warped_binary.shape
        horizontal = cv2.morphologyEx(
            warped_binary,
            cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_RECT, (max(20, width // 3), 1)),
        )
        vertical = cv2.morphologyEx(
            warped_binary,
            cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(15, height // 4))),
        )
        # Bridge the gaps a slightly skewed separator leaves in the projection.
        horizontal = cv2.dilate(
            horizontal, cv2.getStructuringElement(cv2.MORPH_RECT, (max(20, width // 6), 1))
        )
        vertical = cv2.dilate(
            vertical, cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(15, height // 6)))
        )
        rows = self._separator_positions(horizontal, axis=0)
        columns = self._separator_positions(vertical, axis=1)
        return rows, columns, horizontal, vertical

    def _row_bounds(self, separators: list[int], height: int) -> tuple[list[tuple[int, int]], bool]:
        """Return the (top, bottom) pixel bounds of each student row.

        The sheet has one header row followed by one row per student, so the
        student table is a run of ``len(students) + 2`` horizontal separators
        whose data-row gaps are equal. More separators than that are normal: on a
        tilted photograph the line morphology can merge the small date table
        above into the same contour, and the corrected image then contains both
        tables. The correct block is therefore chosen as the candidate window
        with the most uniform data-row spacing rather than assumed to be the
        whole image.

        Only if no such window exists does the method fall back to an equal
        split, which drifts because the header row is shorter than the data rows.
        """
        row_count = len(self.course.students)
        window_size = row_count + 2

        best_window: list[int] | None = None
        best_score = float("inf")
        for start in range(0, len(separators) - window_size + 1):
            window = separators[start : start + window_size]
            gaps = [second - first for first, second in zip(window, window[1:])]
            header_gap, data_gaps = gaps[0], gaps[1:]
            if min(data_gaps) <= 0:
                continue

            average_gap = sum(data_gaps) / len(data_gaps)
            # A data row must be a sensible size, and the header row must be
            # roughly comparable to a data row rather than a whole second table.
            if average_gap < 8 or not 0.4 <= header_gap / average_gap <= 1.8:
                continue

            spread = max(data_gaps) - min(data_gaps)
            score = spread / average_gap
            if score < best_score - 1e-9:
                best_score, best_window = score, window

        if best_window is not None:
            return (
                [(best_window[i + 1], best_window[i + 2]) for i in range(row_count)],
                True,
            )

        fitted = self._fit_row_pitch(separators, height, row_count)
        if fitted is not None:
            return fitted, True

        band = height / (row_count + 1)
        return (
            [
                (int(round(band * (index + 1))), int(round(band * (index + 2))))
                for index in range(row_count)
            ],
            False,
        )

    @staticmethod
    def _fit_row_pitch(
        separators: list[int], height: int, row_count: int
    ) -> list[tuple[int, int]] | None:
        """Recover the row boundaries when some printed rules are too faint to detect.

        The data rows of a signing sheet are equally spaced, so the boundaries
        form an arithmetic sequence. When a faint rule is missed the exact-window
        search above finds nothing, but the sequence can still be recovered from
        the rules that were seen: the pitch is the median observed gap, and the
        offset is the anchor whose sequence best explains them.

        The fit is scored against the separators that were *observed*, not
        against the model points, so a missing rule costs nothing while a rule
        that contradicts the model does.
        """
        if len(separators) < 4:
            return None
        gaps = [b - a for a, b in zip(separators, separators[1:]) if b - a > 8]
        if not gaps:
            return None
        pitch = float(np.median(gaps))

                # The table holds one header band and one band per student, so a
        # plausible pitch is close to height / (students + 1). This rejects a
        # sequence fitted to unrelated marks.
        expected_pitch = height / (row_count + 1)
        if not 0.70 * expected_pitch <= pitch <= 1.35 * expected_pitch:
            return None

        best_model: list[float] | None = None
        best_error = float("inf")
        best_support = 0
        for anchor in separators:
            model = [anchor + step * pitch for step in range(row_count + 1)]
            if model[0] < -2 or model[-1] > height * 1.15:
                continue
            observed = [
                s for s in separators
                if model[0] - 0.6 * pitch <= s <= model[-1] + 0.6 * pitch
            ]
            if len(observed) < 4:
                continue
            # A model point counts as supported when a printed rule was actually
            # seen near it. Requiring several supported points stops a sequence
            # being fitted through noise.
            supported = sum(
                1 for m in model if min(abs(m - s) for s in separators) <= 0.25 * pitch
            )
            if supported < 4:
                continue
            error = sum(min(abs(m - s) for m in model) for s in observed) / len(observed)
            if error < best_error or (error == best_error and supported > best_support):
                best_error, best_model, best_support = error, model, supported

        if best_model is None or best_error >= 0.18 * pitch:
            return None
        return [
            (int(round(best_model[i])), int(round(min(best_model[i + 1], height))))
            for i in range(row_count)
        ]

    def _signature_column(
        self, separators: list[int], width: int
    ) -> tuple[tuple[int, int], bool]:
        """Return the (left, right) pixel bounds of the signature column."""
        if len(separators) >= 2:
            left, right = separators[-2], separators[-1]
            wide_enough = (right - left) > 0.08 * width
            rightmost = right > 0.9 * width
            if wide_enough and rightmost:
                return (left, right), True

        left = int(round(width * self.course.signature_column_start))
        return (left, width - 1), False

        