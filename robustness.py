#!/usr/bin/env python3
"""Measure how far the pipeline can be pushed before it stops being correct.

    python robustness.py

Matching the manual labels on five clean photographs says very little on its
own. This program re-runs the same five sheets through a set of synthetic
degradations - rotation, rescaling, blur, exposure, JPEG compression and sensor
noise - and reports the cell accuracy of each one. The results are written to
``output/robustness.txt`` and are the evidence for the sensitivity table in the
report.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import tempfile

import cv2
import numpy as np

from attendance.image_processor import AttendanceImageProcessor
from attendance.xml_loader import load_course_info

#: Manually read ground truth, in info.xml student order. The last cell of
#: sheet 2 is absent because staff struck an "ab" mark through it, not because
#: the cell is empty.
EXPECTED = {
    "sheet1.jpeg": [True, True, True, True, True, True],
    "sheet2.jpeg": [True, True, True, True, True, False],
    "sheet3.jpeg": [True, False, False, True, True, True],
    "sheet4.jpeg": [True, False, True, False, True, True],
    "sheet5.jpeg": [True, True, True, True, True, True],
}


def rotate(image: np.ndarray, degrees: float) -> np.ndarray:
    height, width = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((width / 2, height / 2), degrees, 1.0)
    cosine, sine = abs(matrix[0, 0]), abs(matrix[0, 1])
    new_width = int(height * sine + width * cosine)
    new_height = int(height * cosine + width * sine)
    matrix[0, 2] += new_width / 2 - width / 2
    matrix[1, 2] += new_height / 2 - height / 2
    return cv2.warpAffine(image, matrix, (new_width, new_height), borderValue=(238, 236, 232))


def scale(image: np.ndarray, factor: float) -> np.ndarray:
    return cv2.resize(image, None, fx=factor, fy=factor, interpolation=cv2.INTER_AREA)


def exposure(image: np.ndarray, factor: float) -> np.ndarray:
    return np.clip(image.astype(np.float32) * factor, 0, 255).astype(np.uint8)


def noise(image: np.ndarray, sigma: float) -> np.ndarray:
    generator = np.random.RandomState(0)
    noisy = image.astype(np.float32) + generator.normal(0, sigma, image.shape)
    return np.clip(noisy, 0, 255).astype(np.uint8)


def perspective(image: np.ndarray, strength: float) -> np.ndarray:
    """Tilt the page as if the phone were held off to one side."""
    height, width = image.shape[:2]
    shift = width * strength
    source = np.float32([[0, 0], [width, 0], [width, height], [0, height]])
    destination = np.float32(
        [[shift, 0], [width, shift * 0.6], [width - shift, height], [0, height - shift * 0.6]]
    )
    matrix = cv2.getPerspectiveTransform(source, destination)
    return cv2.warpPerspective(image, matrix, (width, height), borderValue=(238, 236, 232))


#: name -> (function applied to the photograph, JPEG quality or None).
#: The baseline is handled separately: the supplied file is used byte for byte,
#: because decoding and re-encoding it is itself a degradation.
PERTURBATIONS: dict[str, tuple] = {
    "None (baseline)": (None, None),
    "JPEG re-encode q95": (lambda i: i, 95),
    "Rotation -10 deg": (lambda i: rotate(i, -10), None),
    "Rotation -5 deg": (lambda i: rotate(i, -5), None),
    "Rotation +5 deg": (lambda i: rotate(i, 5), None),
    "Rotation +10 deg": (lambda i: rotate(i, 10), None),
    "Perspective tilt 8%": (lambda i: perspective(i, 0.08), None),
    "Perspective tilt 15%": (lambda i: perspective(i, 0.15), None),
    "Downscale to 50%": (lambda i: scale(i, 0.50), None),
    "Downscale to 30%": (lambda i: scale(i, 0.30), None),
    "Downscale to 20%": (lambda i: scale(i, 0.20), None),
    "Gaussian blur 7 px": (lambda i: cv2.GaussianBlur(i, (7, 7), 0), None),
    "Gaussian blur 15 px": (lambda i: cv2.GaussianBlur(i, (15, 15), 0), None),
    "Over-exposure +30%": (lambda i: exposure(i, 1.30), None),
    "Under-exposure -35%": (lambda i: exposure(i, 0.65), None),
    "Sensor noise sigma 15": (lambda i: noise(i, 15), None),
    "JPEG quality 30": (lambda i: i, 30),
    "JPEG quality 10": (lambda i: i, 10),
}


def evaluate(processor, course, images_dir: Path, workspace: Path) -> list[tuple]:
    rows = []
    for name, (transform, quality) in PERTURBATIONS.items():
        correct = total = sheets_ok = 0
        failures: list[str] = []
        for session in course.sessions:
            source = images_dir / session.image
            if transform is None:
                candidate = source
            else:
                original = cv2.imread(str(source))
                candidate = workspace / f"{name.replace(' ', '_').replace('%', '')}_{session.image}"
                if quality is None:
                    cv2.imwrite(str(candidate), transform(original))
                else:
                    cv2.imwrite(
                        str(candidate), transform(original), [cv2.IMWRITE_JPEG_QUALITY, quality]
                    )

            expected = EXPECTED[session.image]
            total += len(expected)
            try:
                result = processor.process(
                    candidate, session.date, output_root=workspace / "out"
                )
                detected = [row.present for row in result.rows]
                correct += sum(a == b for a, b in zip(detected, expected))
                if detected == expected:
                    sheets_ok += 1
                else:
                    failures.append(f"{session.image} misread")
            except Exception as error:  # noqa: BLE001 - this is the measurement
                failures.append(f"{session.image} {type(error).__name__}")

        rows.append((name, correct, total, sheets_ok, len(course.sessions), failures))
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--info", default="info.xml")
    parser.add_argument("--images", default="input_signing_sheets")
    parser.add_argument("--output", default="output/robustness.txt")
    arguments = parser.parse_args()

    try:
        course = load_course_info(arguments.info)
        processor = AttendanceImageProcessor(course)
        with tempfile.TemporaryDirectory() as workspace:
            rows = evaluate(processor, course, Path(arguments.images), Path(workspace))

        lines = [
            "Robustness of the attendance classifier under synthetic degradation",
            "Each condition re-processes all five supplied sheets (30 signature cells).",
            "",
            f"{'Condition':<24}{'Cells correct':>15}{'Accuracy':>10}{'Sheets exact':>14}  Notes",
            "-" * 96,
        ]
        for name, correct, total, sheets_ok, sheets, failures in rows:
            accuracy = 100.0 * correct / total if total else 0.0
            note = "; ".join(failures) if failures else "-"
            lines.append(
                f"{name:<24}{f'{correct}/{total}':>15}{accuracy:>9.1f}%"
                f"{f'{sheets_ok}/{sheets}':>14}  {note}"
            )
        report = "\n".join(lines)
        print(report)

        output = Path(arguments.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(report + "\n", encoding="utf-8")

        # A machine-readable copy so that the report figure is generated from the
        # measurements rather than by parsing the formatted table back again.
        data = output.with_suffix(".json")
        data.write_text(
            json.dumps(
                [
                    {
                        "condition": name,
                        "cells_correct": correct,
                        "cells_total": total,
                        "accuracy": round(100.0 * correct / total, 1) if total else 0.0,
                        "sheets_exact": sheets_ok,
                        "sheets_total": sheets,
                        "failures": failures,
                    }
                    for name, correct, total, sheets_ok, sheets, failures in rows
                ],
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"\nSaved to: {output} and {data}")
        return 0
    except Exception as error:  # noqa: BLE001 - CLI boundary
        print(f"ERROR: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
