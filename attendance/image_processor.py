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
        # grid detection
        # 1. Separator Position Detection 

    @staticmethod
    def _separator_positions(line_mask: np.ndarray, axis: int) -> list[int]:
        """Return the centre position of every printed separator in ``line_mask``.

        ``axis=0`` projects onto the rows and therefore returns the y positions of
        horizontal separators; ``axis=1`` returns the x positions of vertical ones.
        """

        # Calculate how much of each row or column contains a detected line.
        profile = line_mask.sum(axis=1 if axis == 0 else 0) / 255.0

        # Get the total length of the row or column.
        span = line_mask.shape[1] if axis == 0 else line_mask.shape[0]

        # Mark positions where enough of the line is filled.
        filled = profile > SEPARATOR_FILL_RATIO * span

        positions: list[int] = []
        current_run: list[int] = []

        # Group consecutive filled pixels into one separator.
        for index, is_filled in enumerate(filled):
            if is_filled:
                current_run.append(index)
            elif current_run:
                # Calculate the centre of the detected separator.
                positions.append(int(round(float(np.mean(current_run)))))
                current_run = []
        # Handle a separator that continues until the end.        
        if current_run:
            positions.append(int(round(float(np.mean(current_run)))))
        return positions

    #  2. Grid Detection 

    def _detect_grid(
        self, warped_binary: np.ndarray
    ) -> tuple[list[int], list[int], np.ndarray, np.ndarray]:
        """Detect the horizontal and vertical separators of the corrected table."""

        # Get the height and width of the table image.
        height, width = warped_binary.shape

        # Extract horizontal table lines using morphological opening.
        horizontal = cv2.morphologyEx(
            warped_binary,
            cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_RECT, (max(20, width // 3), 1)),
        )

        # Extract vertical table lines.
        vertical = cv2.morphologyEx(
            warped_binary,
            cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(15, height // 4))),
        )

        # Connect small gaps in horizontal lines.
        horizontal = cv2.dilate(
            horizontal, cv2.getStructuringElement(cv2.MORPH_RECT, (max(20, width // 6), 1))
        )

        # Connect small gaps in vertical lines.
        vertical = cv2.dilate(
            vertical, cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(15, height // 6)))
        )

        # Find the y-coordinates of horizontal separators.
        rows = self._separator_positions(horizontal, axis=0)

        # Find the x-coordinates of vertical separators.
        columns = self._separator_positions(vertical, axis=1)
        return rows, columns, horizontal, vertical

    # ========================== 3. Student Row Detection ==========================

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
        # Number of students in the attendance sheet.
        row_count = len(self.course.students)

        # One header row + student rows require this number of separators.
        window_size = row_count + 2

        best_window: list[int] | None = None
        best_score = float("inf")

        # Check different groups of separators to find the
        # most uniform student-row spacing.
        for start in range(0, len(separators) - window_size + 1):

            # Select a possible table section.
            window = separators[start : start + window_size]

            # Calculate the distance between neighbouring separators.
            gaps = [second - first for first, second in zip(window, window[1:])]
            header_gap, data_gaps = gaps[0], gaps[1:]

            # Ignore invalid row sizes.
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

    #  4. Recover Missing Row Lines 

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

    #  5. Signature Column Detection 

    def _signature_column(
        self, separators: list[int], width: int
    ) -> tuple[tuple[int, int], bool]:
        """Return the (left, right) pixel bounds of the signature column."""

    # If enough vertical separators are detected,
    # use the last two separators as the signature column.
        if len(separators) >= 2:
            left, right = separators[-2], separators[-1]
            wide_enough = (right - left) > 0.08 * width
            rightmost = right > 0.9 * width
            if wide_enough and rightmost:
                return (left, right), True

        left = int(round(width * self.course.signature_column_start))
        return (left, width - 1), False

    #  6. Remove Table Grid Lines 

    @staticmethod
    def _remove_grid_lines(binary_roi: np.ndarray) -> np.ndarray:
        """Delete printed rules and speckle noise, leaving only handwriting."""
        horizontal_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (max(15, binary_roi.shape[1] // 3), 1)
        )

        # Create a vertical kernel for detecting vertical lines.
        vertical_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (1, max(10, binary_roi.shape[0] // 2))
        )
        horizontal = cv2.morphologyEx(binary_roi, cv2.MORPH_OPEN, horizontal_kernel)
        vertical = cv2.morphologyEx(binary_roi, cv2.MORPH_OPEN, vertical_kernel)
        lines = cv2.bitwise_or(horizontal, vertical)
        residual = cv2.bitwise_and(binary_roi, cv2.bitwise_not(lines))

        component_count, labels, statistics, _ = cv2.connectedComponentsWithStats(residual)
        cleaned = np.zeros_like(residual)
        for component in range(1, component_count):
            area = statistics[component, cv2.CC_STAT_AREA]
            component_width = statistics[component, cv2.CC_STAT_WIDTH]
            component_height = statistics[component, cv2.CC_STAT_HEIGHT]
            if area >= 6 and (component_width >= 3 or component_height >= 3):
                cleaned[labels == component] = 255
        return cleaned

    # 7. Detect Coloured Pen Marks
    @staticmethod
    def _colour_mask(roi: np.ndarray) -> np.ndarray:
        """Mask the saturated, non-white pixels produced by a coloured pen."""
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        saturation = hsv[:, :, 1]
        value = hsv[:, :, 2]
        return np.where((saturation > 45) & (value < 245), 255, 0).astype(np.uint8)

    # 8. Detect Absence Annotation
    def is_absence_annotation(
        self,
        mask: np.ndarray,
        max_overlap: float = 0.35,
        level_tolerance: float = 0.30,
    ) -> bool:
        """True when the cell holds a struck-out absence mark rather than a signature.

        Staff record a known absence by writing a short token such as ``ab`` and
        striking a rule through it on both sides, so the cell contains ink and
        the ink-ratio rule of :meth:`_classify` would call it present.

        The mark is recognised by its geometry rather than by reading it. A
        strike stroke is a component that is long and thin, and in a struck-out
        mark it sits *beside* the written token at the same height. That is what
        separates it from a signature that happens to be underlined, where the
        rule sits *below* the writing and therefore overlaps it horizontally.
        """
        height, width = mask.shape
        # Compression and rescaling break a thin strike stroke into fragments,
        # none of which is long enough to be recognised. Bridging the gaps first
        # removed the one false positive this rule produced across the
        # degradation suite, at no cost to the number it detects.
        bridged = cv2.dilate(
            mask, cv2.getStructuringElement(cv2.MORPH_RECT, (STRIKE_BRIDGE, 1))
        )
        count, _, statistics, _ = cv2.connectedComponentsWithStats(bridged)

        strikes: list[int] = []
        token: list[int] = []
        for component in range(1, count):
            component_width = statistics[component, cv2.CC_STAT_WIDTH]
            component_height = statistics[component, cv2.CC_STAT_HEIGHT]
            long_and_thin = (
                component_width >= self.course.strike_min_width * width
                and component_height <= self.course.strike_max_height * height
            )
            (strikes if long_and_thin else token).append(component)

        if not strikes or not token:
            return False

        written = max(token, key=lambda c: statistics[c, cv2.CC_STAT_AREA])
        token_left = statistics[written, cv2.CC_STAT_LEFT]
        token_width = statistics[written, cv2.CC_STAT_WIDTH]
        token_centre = statistics[written, cv2.CC_STAT_TOP] + statistics[
            written, cv2.CC_STAT_HEIGHT
        ] / 2

        for strike in strikes:
            strike_left = statistics[strike, cv2.CC_STAT_LEFT]
            strike_width = statistics[strike, cv2.CC_STAT_WIDTH]
            strike_centre = statistics[strike, cv2.CC_STAT_TOP] + statistics[
                strike, cv2.CC_STAT_HEIGHT
            ] / 2

            shared = max(
                0,
                min(token_left + token_width, strike_left + strike_width)
                - max(token_left, strike_left),
            )
            beside = shared / max(min(token_width, strike_width), 1) <= max_overlap
            same_level = abs(strike_centre - token_centre) <= level_tolerance * height
            if beside and same_level:
                return True
        return False

    # 9. Final Attendance Classification

    def _classify(self, colour_ratio: float, residual_ratio: float) -> tuple[bool, float]:
        """Apply the dual-evidence decision rule and derive a confidence value."""
        colour_evidence = colour_ratio / self.course.color_ratio_threshold
        residual_evidence = residual_ratio / self.course.residual_ratio_threshold
        evidence = max(colour_evidence, residual_evidence)
        present = evidence >= 1.0

        if present:
            confidence = 0.55 + 0.20 * min(evidence, 2.2)
        else:
            confidence = 0.95 - 0.30 * evidence
        return present, float(np.clip(confidence, 0.55, 0.99))
def process(
        self,
        image_path: str | Path,
        session_date: str,
        output_root: str | Path = "output",
        progress: ProgressCallback | None = None,
    ) -> ProcessingResult:
        """Run the complete pipeline for one signing-sheet photograph."""
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        output_dir = Path(output_root) / image_path.stem
        output_dir.mkdir(parents=True, exist_ok=True)
        notify = progress or (lambda message: None)

        notify("1/9 Loading and resizing image")
        original = cv2.imread(str(image_path))
        if original is None:
            raise ValueError(f"Unsupported or damaged image: {image_path}")
        scale = self.target_width / original.shape[1]
        resized = cv2.resize(
            original, (self.target_width, int(original.shape[0] * scale))
        )
        self._save(output_dir / "01_original_resized.jpg", resized)

        notify("2/9 Converting to greyscale")
        greyscale = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        self._save(output_dir / "02_greyscale.png", greyscale)

        notify("3/9 Applying adaptive binarization")
        binary = self._binarize(greyscale)
        self._save(output_dir / "03_binary.png", binary)

        notify("4/9 Estimating and correcting page skew")
        skew_angle = self.estimate_skew(binary)
        if abs(skew_angle) >= 0.4:
            rotated = self.rotate_upright(resized, skew_angle)
            # Rotation grows the canvas, so rescale to keep every kernel size and
            # every relative measurement below on the same footing as before.
            rotation_scale = self.target_width / rotated.shape[1]
            resized = cv2.resize(
                rotated,
                (self.target_width, max(1, int(rotated.shape[0] * rotation_scale))),
            )
            binary = self._binarize(resized)
            self._save(output_dir / "03b_deskewed.jpg", resized)
            self._save(output_dir / "03c_deskewed_binary.png", binary)
        notify(f"    estimated skew: {skew_angle:+.2f} degrees")

        notify("5/9 Detecting table lines and correcting perspective")
        warped, horizontal, vertical, quadrilateral = self._find_student_table(
            resized, binary
        )
        self._save(output_dir / "04_horizontal_lines.png", horizontal)
        self._save(output_dir / "05_vertical_lines.png", vertical)
        detected = resized.copy()
        cv2.polylines(detected, [quadrilateral.astype(np.int32)], True, (0, 200, 0), 5)
        self._save(output_dir / "06_detected_student_table.jpg", detected)
        self._save(output_dir / "07_perspective_corrected_table.jpg", warped)

        notify("6/9 Locating the printed row and column separators")
        warped_binary = self._binarize(warped)
        height, width = warped_binary.shape
        row_separators, column_separators, warped_horizontal, _ = self._detect_grid(
            warped_binary
        )
        rows, grid_rows_detected = self._row_bounds(row_separators, height)
        (column_start, column_end), grid_column_detected = self._signature_column(
            column_separators, width
        )
        grid_detected = grid_rows_detected and grid_column_detected

        grid_preview = warped.copy()
        for position in row_separators:
            cv2.line(grid_preview, (0, position), (width, position), (0, 140, 255), 2)
        for position in column_separators:
            cv2.line(grid_preview, (position, 0), (position, height), (255, 120, 0), 2)
        self._save(output_dir / "08_detected_grid.jpg", grid_preview)
        self._save(output_dir / "09_warped_horizontal_lines.png", warped_horizontal)

        notify("7/9 Extracting student signature cells")
        margin_x = max(3, int(0.03 * (column_end - column_start)))
        row_margin = max(2, int(0.10 * (height / (len(self.course.students) + 1))))
        cell_left = column_start + margin_x
        cell_right = column_end - margin_x

        row_results: list[RowDetection] = []
        combined_mask = np.zeros((height, width), dtype=np.uint8)
        annotated = warped.copy()

        for student, (row_top, row_bottom) in zip(self.course.students, rows):
            top = row_top + row_margin
            bottom = row_bottom - row_margin
            roi = warped[top:bottom, cell_left:cell_right]
            if roi.size == 0:
                raise TableDetectionError(
                    f"The signature cell for {student.index} was empty after table extraction"
                )

            colour_mask = self._colour_mask(roi)
            dark_ink = cv2.adaptiveThreshold(
                cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY),
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV,
                21,
                9,
            )
            residual_mask = self._remove_grid_lines(dark_ink)
            merged_mask = cv2.bitwise_or(colour_mask, residual_mask)

            cell_area = float(roi.shape[0] * roi.shape[1])
            colour_pixels = int(cv2.countNonZero(colour_mask))
            residual_pixels = int(cv2.countNonZero(residual_mask))
            colour_ratio = colour_pixels / cell_area
            residual_ratio = residual_pixels / cell_area
            present, confidence = self._classify(colour_ratio, residual_ratio)
            reason = "signature" if present else "blank cell"
            if present and self.is_absence_annotation(merged_mask):
                # The cell holds ink, but it is a struck-out absence mark, not a
                # signature, so the ink-ratio decision has to be overruled.
                present, confidence, reason = False, 0.95, "absence annotation"

            signature_file = output_dir / "signatures" / f"{student.index}.png"
            self._save(signature_file, merged_mask)
            crop_file = output_dir / "signature_crops" / f"{student.index}.png"
            self._save(crop_file, roi)
            combined_mask[top:bottom, cell_left:cell_right] = merged_mask

            box_colour = (0, 170, 0) if present else (0, 0, 220)
            cv2.rectangle(annotated, (cell_left, top), (cell_right, bottom), box_colour, 2)

            row_results.append(
                RowDetection(
                    student=student,
                    present=present,
                    confidence=confidence,
                    color_pixels=colour_pixels,
                    residual_pixels=residual_pixels,
                    color_ratio=colour_ratio,
                    residual_ratio=residual_ratio,
                    signature_path=signature_file,
                    crop_path=crop_file,
                    row_top=top,
                    row_bottom=bottom,
                    reason=reason,
                )
            )

        notify("8/9 Classifying attendance")
        self._save(output_dir / "10_combined_signature_mask.png", combined_mask)
        self._save(
            output_dir / "11_attendance_result.jpg",
            self._annotate(annotated, row_results),
        )

        notify("9/9 Processing complete")
        return ProcessingResult(
            image_path=image_path,
            session_date=session_date,
            output_dir=output_dir,
            rows=row_results,
            grid_detected=grid_detected,
            table_size=(width, height),
            skew_angle=skew_angle,
        )
    @staticmethod
    def _annotate(table: np.ndarray, rows: list[RowDetection]) -> np.ndarray:
        """Add a status strip to the left of the table instead of drawing over it."""
        strip_width = 150
        height = table.shape[0]
        canvas = np.full((height, table.shape[1] + strip_width, 3), 255, dtype=np.uint8)
        canvas[:, strip_width:] = table

        for row in rows:
            status = "PRESENT" if row.present else "ABSENT"
            colour = (0, 150, 0) if row.present else (0, 0, 210)
            centre = (row.row_top + row.row_bottom) // 2
            cv2.putText(
                canvas,
                status,
                (10, centre + 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.62,
                colour,
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                canvas,
                f"{row.confidence:.0%}",
                (10, centre + 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (90, 90, 90),
                1,
                cv2.LINE_AA,
            )
        return canvas

        

    


        