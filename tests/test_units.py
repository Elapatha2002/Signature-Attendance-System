"""Unit tests for the XML loader, database, classifier rule and signature matcher."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from attendance.database import AttendanceDatabase
from attendance.image_processor import AttendanceImageProcessor, TableDetectionError
from attendance.models import CourseInfo, ProcessingResult, RowDetection, Session, Student, Subject
from attendance.signature_matcher import (
    calibrate_threshold,
    collect_samples,
    compare_signatures,
    normalize_signature,
)
from attendance.xml_loader import date_for_image, load_course_info


# ------------------------------------------------------------------ XML input


def test_xml_loads_expected_records(course) -> None:
    assert course.subject.code == "CS402.3"
    assert len(course.students) == 6
    assert len(course.sessions) == 5
    assert course.students[0].index == "10000409"


def test_xml_thresholds_are_configurable(tmp_path: Path) -> None:
    xml = tmp_path / "info.xml"
    xml.write_text(
        """<?xml version="1.0"?>
        <attendanceSystem>
          <subject><code>X</code><title>T</title><lecturer>L</lecturer></subject>
          <students><student index="1" title="Mr">A B</student></students>
          <sessions><session image="a.jpg" date="01/01/2020"/></sessions>
          <processing signatureColumnStart="0.5" colorRatioThreshold="0.5"
                      residualRatioThreshold="0.4"/>
        </attendanceSystem>""",
        encoding="utf-8",
    )
    course = load_course_info(xml)
    assert course.signature_column_start == 0.5
    assert course.color_ratio_threshold == 0.5
    assert course.residual_ratio_threshold == 0.4


def test_xml_rejects_a_student_without_an_index(tmp_path: Path) -> None:
    xml = tmp_path / "bad.xml"
    xml.write_text(
        """<?xml version="1.0"?>
        <attendanceSystem>
          <subject><code>X</code><title>T</title><lecturer>L</lecturer></subject>
          <students><student title="Mr">No index</student></students>
        </attendanceSystem>""",
        encoding="utf-8",
    )
    with pytest.raises(ValueError):
        load_course_info(xml)


def test_missing_xml_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_course_info(tmp_path / "nothing.xml")


def test_date_lookup_is_case_insensitive(course) -> None:
    assert date_for_image(course, "SHEET3.JPEG") == "28/06/2019"
    assert date_for_image(course, "unknown.jpeg") is None


# ------------------------------------------------------------------- database


def _tiny_course() -> CourseInfo:
    return CourseInfo(
        subject=Subject(code="CS402.3", title="CGV", lecturer="Dr R"),
        students=[Student(index="001", title="Mr", name="A B"), Student(index="002", title="Ms", name="C D")],
        sessions=[Session(image="a.jpg", date="01/01/2020")],
    )


def _result(date: str, present: tuple[bool, bool], course: CourseInfo) -> ProcessingResult:
    return ProcessingResult(
        image_path=Path(f"{date.replace('/', '-')}.jpg"),
        session_date=date,
        output_dir=Path("."),
        rows=[
            RowDetection(
                student=student,
                present=flag,
                confidence=0.9,
                color_pixels=10,
                residual_pixels=5,
                color_ratio=0.1,
                residual_ratio=0.05,
                signature_path=Path("sig.png"),
            )
            for student, flag in zip(course.students, present)
        ],
    )


def test_database_round_trip(tmp_path: Path) -> None:
    course = _tiny_course()
    with AttendanceDatabase(tmp_path / "a.db") as database:
        database.save_result(course, _result("01/01/2020", (True, False), course))
        database.save_result(course, _result("02/01/2020", (True, True), course))

        assert len(database.all_results()) == 4
        summary = {row["student_index"]: row for row in database.class_summary()}
        assert summary["001"]["present_count"] == 2
        assert summary["002"]["present_count"] == 1
        assert summary["002"]["rate"] == 50.0


def test_reprocessing_updates_instead_of_duplicating(tmp_path: Path) -> None:
    """Re-running a session must overwrite its rows, not attach them elsewhere."""
    course = _tiny_course()
    with AttendanceDatabase(tmp_path / "b.db") as database:
        database.save_result(course, _result("01/01/2020", (True, True), course))
        database.save_result(course, _result("01/01/2020", (False, False), course))

        rows = database.all_results()
        assert len(rows) == 2
        assert {row["status"] for row in rows} == {"Absent"}


def test_session_id_is_correct_after_a_new_student_is_added(tmp_path: Path) -> None:
    """Guards the upsert: seed() inserting a student must not steal lastrowid."""
    course = _tiny_course()
    database_path = tmp_path / "c.db"
    with AttendanceDatabase(database_path) as database:
        database.save_result(course, _result("01/01/2020", (True, True), course))

    extended = CourseInfo(
        subject=course.subject,
        students=[*course.students, Student(index="003", title="Mr", name="E F")],
        sessions=course.sessions,
    )
    with AttendanceDatabase(database_path) as database:
        database.save_result(extended, _result("01/01/2020", (True, True, True), extended))
        sessions = database.session_summary()
        assert len(sessions) == 1
        assert sessions[0]["student_count"] == 3


def test_student_summary_is_ordered_by_real_date(tmp_path: Path) -> None:
    """Dates are stored as DD/MM/YYYY, so string ordering would be wrong."""
    course = _tiny_course()
    with AttendanceDatabase(tmp_path / "d.db") as database:
        for date in ("31/05/2019", "05/07/2019", "21/06/2019"):
            database.save_result(course, _result(date, (True, True), course))
        dates = [row["session_date"] for row in database.student_summary("001")]
        assert dates == ["31/05/2019", "21/06/2019", "05/07/2019"]


def test_status_check_constraint(tmp_path: Path) -> None:
    import sqlite3

    course = _tiny_course()
    with AttendanceDatabase(tmp_path / "e.db") as database:
        database.save_result(course, _result("01/01/2020", (True, True), course))
        with pytest.raises(sqlite3.IntegrityError):
            database.connection.execute(
                "UPDATE attendance SET status='Maybe' WHERE student_index='001'"
            )


# ------------------------------------------------------------------ processor


def test_classification_rule_is_an_or_over_both_kinds_of_ink(course) -> None:
    processor = AttendanceImageProcessor(course)
    colour, residual = course.color_ratio_threshold, course.residual_ratio_threshold

    assert processor._classify(colour, 0.0)[0] is True
    assert processor._classify(0.0, residual)[0] is True
    assert processor._classify(colour * 0.99, residual * 0.99)[0] is False
    assert processor._classify(0.0, 0.0)[0] is False


def test_confidence_rises_with_evidence(course) -> None:
    processor = AttendanceImageProcessor(course)
    weak = processor._classify(course.color_ratio_threshold * 1.05, 0.0)[1]
    strong = processor._classify(course.color_ratio_threshold * 8, 0.0)[1]
    assert strong > weak


def test_absence_annotation_is_recognised_by_its_strike_through(course) -> None:
    processor = AttendanceImageProcessor(course)
    mask = np.zeros((60, 240), np.uint8)
    cv2.putText(mask, "ab", (100, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, 255, 3)
    cv2.line(mask, (10, 30), (90, 30), 255, 2)     # strike to the left
    cv2.line(mask, (160, 30), (230, 30), 255, 2)   # strike to the right
    assert processor.is_absence_annotation(mask) is True


def test_an_underlined_signature_is_not_an_absence_annotation(course) -> None:
    """The rule must key on a strike beside the writing, not a rule below it."""
    processor = AttendanceImageProcessor(course)
    mask = np.zeros((60, 240), np.uint8)
    cv2.putText(mask, "Sig", (30, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.2, 255, 3)
    cv2.line(mask, (20, 50), (210, 50), 255, 2)    # underline below the writing
    assert processor.is_absence_annotation(mask) is False


def test_a_plain_signature_is_not_an_absence_annotation(course) -> None:
    processor = AttendanceImageProcessor(course)
    mask = np.zeros((60, 240), np.uint8)
    cv2.putText(mask, "Perera", (20, 42), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 1.2, 255, 3)
    assert processor.is_absence_annotation(mask) is False


def test_an_empty_cell_is_not_an_absence_annotation(course) -> None:
    processor = AttendanceImageProcessor(course)
    assert processor.is_absence_annotation(np.zeros((60, 240), np.uint8)) is False


def test_strike_geometry_is_configurable(course) -> None:
    from dataclasses import replace

    mask = np.zeros((60, 240), np.uint8)
    cv2.putText(mask, "ab", (100, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, 255, 3)
    cv2.line(mask, (10, 30), (90, 30), 255, 2)
    cv2.line(mask, (160, 30), (230, 30), 255, 2)

    assert AttendanceImageProcessor(course).is_absence_annotation(mask) is True
    # Demanding a strike wider than the whole cell must switch the rule off.
    strict = replace(course, strike_min_width=1.5)
    assert AttendanceImageProcessor(strict).is_absence_annotation(mask) is False


def test_corner_ordering(course) -> None:
    processor = AttendanceImageProcessor(course)
    scrambled = np.array([[10, 90], [90, 10], [10, 10], [90, 90]], dtype=np.float32)
    ordered = processor._order_points(scrambled)
    assert ordered.tolist() == [[10, 10], [90, 10], [90, 90], [10, 90]]


def test_row_bounds_fall_back_when_the_grid_is_not_found(course) -> None:
    processor = AttendanceImageProcessor(course)
    bounds, detected = processor._row_bounds([], 350)
    assert detected is False
    assert len(bounds) == len(course.students)

    separators = [0, 40, 90, 140, 190, 240, 290, 340]
    bounds, detected = processor._row_bounds(separators, 350)
    assert detected is True
    assert bounds[0] == (40, 90)
    assert bounds[-1] == (290, 340)


def test_row_bounds_ignore_a_merged_date_table(course) -> None:
    """A tilted photograph can merge the date table into the same contour.

    The student block must still be picked out by its uniform row spacing
    instead of the rows being spread across both tables.
    """
    processor = AttendanceImageProcessor(course)
    # Date table (two rows), then the student table (header + six rows).
    separators = [3, 42, 111, 134, 173, 220, 267, 314, 362, 408, 456]
    bounds, detected = processor._row_bounds(separators, 464)
    assert detected is True
    assert bounds[0] == (173, 220)
    assert bounds[-1] == (408, 456)


def test_row_bounds_reject_a_window_whose_header_is_a_whole_table(course) -> None:
    processor = AttendanceImageProcessor(course)
    # A 200 px "header" in front of 40 px rows is another table, not a header.
    separators = [0, 200, 240, 280, 320, 360, 400, 440]
    _, detected = processor._row_bounds(separators, 460)
    assert detected is False


def test_signature_column_falls_back_to_the_xml_fraction(course) -> None:
    processor = AttendanceImageProcessor(course)
    (left, right), detected = processor._signature_column([0, 500], 1200)
    assert detected is False
    assert left == int(1200 * course.signature_column_start)

    (left, right), detected = processor._signature_column([0, 140, 940, 1196], 1200)
    assert detected is True
    assert (left, right) == (940, 1196)


def test_skew_estimation_recovers_an_applied_rotation(course, project_root: Path) -> None:
    processor = AttendanceImageProcessor(course)
    image = cv2.imread(str(project_root / "input_signing_sheets" / "sheet3.jpeg"))
    resized = cv2.resize(image, (1400, int(image.shape[0] * 1400 / image.shape[1])))

    baseline = processor.estimate_skew(processor._binarize(resized))
    for applied in (-8.0, 5.0):
        tilted = processor.rotate_upright(resized, -applied)
        measured = processor.estimate_skew(processor._binarize(tilted))
        assert measured == pytest.approx(baseline + applied, abs=1.5)


def test_rotation_replicates_the_border_instead_of_filling_it(course) -> None:
    """A constant fill would create a page-sized rectangle the grid mistakes for a table."""
    processor = AttendanceImageProcessor(course)
    image = np.full((200, 300, 3), 40, dtype=np.uint8)
    rotated = processor.rotate_upright(image, 12.0)
    assert rotated.shape[0] > 200 and rotated.shape[1] > 300
    assert rotated.max() <= 41, "an artificial bright border was introduced"


def test_contrast_normalisation_survives_under_exposure(course, project_root: Path) -> None:
    processor = AttendanceImageProcessor(course)
    image = cv2.imread(str(project_root / "input_signing_sheets" / "sheet2.jpeg"))
    dark = np.clip(image.astype(np.float32) * 0.65, 0, 255).astype(np.uint8)
    normalised = processor.normalize_contrast(cv2.cvtColor(dark, cv2.COLOR_BGR2GRAY))
    original = cv2.cvtColor(dark, cv2.COLOR_BGR2GRAY)
    assert normalised.std() > original.std()


def test_a_photograph_without_a_table_is_reported_clearly(course, tmp_path: Path) -> None:
    blank = tmp_path / "blank.png"
    cv2.imwrite(str(blank), np.full((900, 700, 3), 255, dtype=np.uint8))
    processor = AttendanceImageProcessor(course)
    with pytest.raises(TableDetectionError):
        processor.process(blank, "01/01/2020", output_root=tmp_path)


def test_missing_image_is_reported_clearly(course, tmp_path: Path) -> None:
    processor = AttendanceImageProcessor(course)
    with pytest.raises(FileNotFoundError):
        processor.process(tmp_path / "missing.jpg", "01/01/2020", output_root=tmp_path)


# ----------------------------------------------------------- signature matcher


def _write_mask(path: Path, draw) -> Path:
    canvas = np.zeros((60, 240), dtype=np.uint8)
    draw(canvas)
    cv2.imwrite(str(path), canvas)
    return path


def test_normalisation_centres_and_scales_the_ink(tmp_path: Path) -> None:
    small = _write_mask(tmp_path / "s.png", lambda c: cv2.line(c, (10, 30), (40, 30), 255, 2))
    large = _write_mask(tmp_path / "l.png", lambda c: cv2.line(c, (20, 10), (220, 50), 255, 3))
    assert normalize_signature(small).shape == normalize_signature(large).shape
    assert normalize_signature(small).any()


def test_normalisation_of_an_empty_mask_is_blank(tmp_path: Path) -> None:
    empty = _write_mask(tmp_path / "e.png", lambda c: None)
    assert not normalize_signature(empty).any()


def test_identical_signatures_score_higher_than_different_ones(tmp_path: Path) -> None:
    first = _write_mask(
        tmp_path / "a.png", lambda c: cv2.putText(c, "AB", (20, 45), 0, 1.4, 255, 3)
    )
    copy = _write_mask(
        tmp_path / "b.png", lambda c: cv2.putText(c, "AB", (60, 45), 0, 1.4, 255, 3)
    )
    other = _write_mask(
        tmp_path / "c.png", lambda c: cv2.circle(c, (120, 30), 22, 255, 3)
    )
    same = compare_signatures(first, copy).combined_score
    different = compare_signatures(first, other).combined_score
    assert same > different


def test_scores_stay_within_range(tmp_path: Path) -> None:
    first = _write_mask(tmp_path / "x.png", lambda c: cv2.line(c, (10, 40), (200, 20), 255, 3))
    second = _write_mask(tmp_path / "y.png", lambda c: cv2.line(c, (10, 20), (200, 40), 255, 3))
    comparison = compare_signatures(first, second)
    for score in (
        comparison.ncc_score,
        comparison.iou_score,
        comparison.profile_score,
        comparison.ssim_score,
        comparison.orb_score,
        comparison.combined_score,
    ):
        assert 0.0 <= score <= 1.0


def test_comparison_is_symmetric(tmp_path: Path) -> None:
    first = _write_mask(tmp_path / "p.png", lambda c: cv2.putText(c, "SR", (20, 45), 0, 1.3, 255, 3))
    second = _write_mask(tmp_path / "q.png", lambda c: cv2.putText(c, "SR", (55, 42), 0, 1.3, 255, 3))
    forward = compare_signatures(first, second).combined_score
    backward = compare_signatures(second, first).combined_score
    assert forward == pytest.approx(backward, abs=1e-6)


def test_collect_samples_skips_absent_and_blank_cells(tmp_path: Path) -> None:
    signed = _write_mask(tmp_path / "signed.png", lambda c: cv2.rectangle(c, (10, 10), (200, 50), 255, -1))
    blank = _write_mask(tmp_path / "blank.png", lambda c: None)
    records = [
        {"student_index": "001", "status": "Present", "signature_path": str(signed)},
        {"student_index": "001", "status": "Absent", "signature_path": str(blank)},
        {"student_index": "002", "status": "Present", "signature_path": str(blank)},
        {"student_index": "003", "status": "Present", "signature_path": None},
    ]
    samples = collect_samples(records)
    assert list(samples) == ["001"]
    assert len(samples["001"]) == 1


def test_consistency_needs_at_least_three_samples(tmp_path: Path) -> None:
    from attendance.signature_matcher import consistency_of_each

    def sample(name: str, offset: int) -> Path:
        return _write_mask(
            tmp_path / name, lambda c: cv2.putText(c, "AB", (20 + offset, 45), 0, 1.4, 255, 3)
        )

    two = [sample("t1.png", 0), sample("t2.png", 20)]
    assert set(consistency_of_each(two).values()) == {None}

    three = two + [sample("t3.png", 40)]
    scores = consistency_of_each(three)
    assert len(scores) == 3
    assert all(0.0 <= value <= 1.0 for value in scores.values())


def test_an_odd_signature_scores_below_its_consistent_siblings(tmp_path: Path) -> None:
    from attendance.signature_matcher import consistency_of_each

    matching = [
        _write_mask(tmp_path / f"m{i}.png",
                    lambda c, i=i: cv2.putText(c, "AB", (20 + 8 * i, 45), 0, 1.4, 255, 3))
        for i in range(3)
    ]
    odd = _write_mask(tmp_path / "odd.png", lambda c: cv2.circle(c, (120, 30), 22, 255, 3))
    scores = consistency_of_each(matching + [odd])
    assert scores[odd] < min(scores[path] for path in matching)


def test_suspicion_rises_as_the_score_falls(tmp_path: Path) -> None:
    from attendance.signature_matcher import Calibration, suspicion_probability

    calibration = Calibration(
        genuine_scores=[0.60, 0.62, 0.65, 0.58, 0.61],
        impostor_scores=[0.30, 0.28, 0.33, 0.31, 0.29],
        threshold=0.45,
    )
    high = suspicion_probability(0.62, calibration)
    low = suspicion_probability(0.30, calibration)
    assert high < 0.1, "a typical genuine score must not look suspicious"
    assert low > 0.9, "a typical impostor score must look suspicious"
    assert 0.0 <= suspicion_probability(0.45, calibration) <= 1.0


def test_suspicion_is_neutral_without_calibration_data() -> None:
    from attendance.signature_matcher import Calibration, suspicion_probability

    empty = Calibration(genuine_scores=[], impostor_scores=[], threshold=0.5)
    assert suspicion_probability(0.5, empty) == 0.5


def test_calibration_places_the_threshold_between_the_two_means(tmp_path: Path) -> None:
    def sample(name: str, text: str, offset: int) -> Path:
        return _write_mask(
            tmp_path / name, lambda c: cv2.putText(c, text, (20 + offset, 45), 0, 1.4, 255, 3)
        )

    samples = {
        "001": [sample("a1.png", "AB", 0), sample("a2.png", "AB", 25)],
        "002": [sample("b1.png", "XY", 0), sample("b2.png", "XY", 25)],
    }
    calibration = calibrate_threshold(samples)
    assert len(calibration.genuine_scores) == 2
    assert len(calibration.impostor_scores) == 4
    assert calibration.impostor_mean <= calibration.threshold <= calibration.genuine_mean
    assert 0.0 <= calibration.roc_auc <= 1.0
