from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class Student:
    index: str
    title: str
    name: str


@dataclass(frozen=True)
class Subject:
    code: str
    title: str
    lecturer: str


@dataclass(frozen=True)
class Session:
    image: str
    date: str


@dataclass(frozen=True)
class CourseInfo:
    subject: Subject
    students: list[Student]
    sessions: list[Session]
    signature_column_start: float = 0.79
    color_ratio_threshold: float = 0.012
    residual_ratio_threshold: float = 0.020
    #: Geometry of the strike-through used to recognise a written absence mark
    #: such as "-ab-", as a fraction of the signature cell.
    strike_min_width: float = 0.22
    strike_max_height: float = 0.22


@dataclass
class RowDetection:
    """Classification evidence for the signature cell of one student row."""

    student: Student
    present: bool
    confidence: float
    color_pixels: int
    residual_pixels: int
    color_ratio: float
    residual_ratio: float
    signature_path: Path | None = None
    #: The signature cell in colour, kept so that an audit can show the
    #: handwriting as staff would see it rather than as a binary mask.
    crop_path: Path | None = None
    row_top: int = 0
    row_bottom: int = 0
    #: Why the decision was made: "signature", "blank cell" or
    #: "absence annotation". Recorded so that a marked-absent cell that
    #: contains ink can be explained.
    reason: str = "signature"

    @property
    def status(self) -> str:
        return "Present" if self.present else "Absent"


@dataclass
class ProcessingResult:
    """Everything one signing-sheet photograph produced."""

    image_path: Path
    session_date: str
    output_dir: Path
    rows: list[RowDetection] = field(default_factory=list)
    #: True when the printed grid was located, so the row and column bounds are
    #: measured rather than estimated from an equal split.
    grid_detected: bool = False
    table_size: tuple[int, int] = (0, 0)
    #: Page rotation removed before table detection, in degrees.
    skew_angle: float = 0.0

    @property
    def present_count(self) -> int:
        return sum(row.present for row in self.rows)
