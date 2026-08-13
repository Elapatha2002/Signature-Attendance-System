"""Regression tests over the five supplied signing sheets.

The expected labels below were produced by manually reading each photograph,
not by recording what the program happened to output.
"""

from __future__ import annotations

import cv2
import pytest

#: Manually read ground truth, in info.xml student order.
EXPECTED = {
    "sheet1.jpeg": [True, True, True, True, True, True],
    "sheet2.jpeg": [True, True, True, True, True, False],
    "sheet3.jpeg": [True, False, False, True, True, True],
    "sheet4.jpeg": [True, False, True, False, True, True],
    "sheet5.jpeg": [True, True, True, True, True, True],
}

#: Cells that are absent because staff wrote a struck-out absence mark into
#: them, rather than because the cell was left blank. These contain ink, so the
#: ink-ratio rule alone would call them present.
ANNOTATED_ABSENCES = {("sheet2.jpeg", "10009306")}

TOTAL_CELLS = 30
EXPECTED_PRESENT = 25


def test_every_sheet_matches_the_manual_labels(course, results) -> None:
    for session in course.sessions:
        detected = [row.present for row in results[session.image].rows]
        assert detected == EXPECTED[session.image], f"mismatch on {session.image}"


def test_dataset_totals(results) -> None:
    cells = sum(len(result.rows) for result in results.values())
    present = sum(result.present_count for result in results.values())
    assert cells == TOTAL_CELLS
    assert present == EXPECTED_PRESENT


def test_printed_grid_is_measured_on_every_sheet(results) -> None:
    """The row and column bounds must come from the printed separators.

    If this fails the pipeline has silently fallen back to an equal-height split,
    which drifts because the header row is shorter than the data rows.
    """
    for image, result in results.items():
        assert result.grid_detected, f"{image} fell back to the estimated grid"


def test_rows_are_ordered_and_do_not_overlap(course, results) -> None:
    for image, result in results.items():
        assert [row.student.index for row in result.rows] == [
            student.index for student in course.students
        ], f"{image} returned rows in the wrong order"
        for previous, current in zip(result.rows, result.rows[1:]):
            assert previous.row_bottom <= current.row_top, f"{image} has overlapping rows"


def test_decisions_have_a_clear_margin(course, results) -> None:
    """Blank and signed cells must be separated by more than a rounding error.

    Matching the labels is not enough on its own: a cell classified correctly
    with an evidence value of 0.99 or 1.01 would be luck rather than a working
    rule.
    """
    present_evidence = []
    blank_evidence = []
    for result in results.values():
        for row in result.rows:
            if row.reason == "absence annotation":
                # Decided by the geometry of the strike-through, not by ink ratio.
                continue
            evidence = max(
                row.color_ratio / course.color_ratio_threshold,
                row.residual_ratio / course.residual_ratio_threshold,
            )
            (present_evidence if row.present else blank_evidence).append(evidence)

    assert min(present_evidence) > 2.0, "a signed cell is uncomfortably close to the threshold"
    assert max(blank_evidence) < 0.8, "a blank cell is uncomfortably close to the threshold"


def test_absent_cells_produce_an_empty_signature_mask(results) -> None:
    for image, result in results.items():
        for row in result.rows:
            if row.present or row.reason == "absence annotation":
                continue
            mask = cv2.imread(str(row.signature_path), cv2.IMREAD_GRAYSCALE)
            assert mask is not None
            ink_ratio = float((mask > 0).mean())
            assert ink_ratio < 0.01, f"{image} {row.student.index} kept ink in a blank cell"


def test_written_absence_marks_are_recognised(results) -> None:
    """A cell struck out with an "ab" mark is absent even though it holds ink."""
    found = set()
    for image, result in results.items():
        for row in result.rows:
            key = (image, row.student.index)
            if row.reason == "absence annotation":
                found.add(key)
                assert not row.present, f"{key} was annotated absent but reported present"
                mask = cv2.imread(str(row.signature_path), cv2.IMREAD_GRAYSCALE)
                assert float((mask > 0).mean()) > 0.02, (
                    f"{key} has no ink, so it is a blank cell rather than an annotation"
                )
    assert found == ANNOTATED_ABSENCES, (
        f"absence annotations detected {sorted(found)}, expected {sorted(ANNOTATED_ABSENCES)}"
    )


def test_every_decision_records_a_reason(results) -> None:
    allowed = {"signature", "blank cell", "absence annotation"}
    for result in results.values():
        for row in result.rows:
            assert row.reason in allowed
            assert (row.reason == "signature") == row.present


def test_present_cells_produce_a_usable_signature_mask(results) -> None:
    for image, result in results.items():
        for row in result.rows:
            if not row.present:
                continue
            mask = cv2.imread(str(row.signature_path), cv2.IMREAD_GRAYSCALE)
            assert mask is not None and mask.any(), f"{image} {row.student.index} mask is empty"


def test_confidence_is_bounded(results) -> None:
    for result in results.values():
        for row in result.rows:
            assert 0.55 <= row.confidence <= 0.99


@pytest.mark.parametrize("image", sorted(EXPECTED))
def test_processing_writes_the_documented_stage_images(results, image) -> None:
    output_dir = results[image].output_dir
    expected_files = [
        "01_original_resized.jpg",
        "02_greyscale.png",
        "03_binary.png",
        "04_horizontal_lines.png",
        "05_vertical_lines.png",
        "06_detected_student_table.jpg",
        "07_perspective_corrected_table.jpg",
        "08_detected_grid.jpg",
        "10_combined_signature_mask.png",
        "11_attendance_result.jpg",
    ]
    missing = [name for name in expected_files if not (output_dir / name).exists()]
    assert not missing, f"{image} did not save {missing}"
