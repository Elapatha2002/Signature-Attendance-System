#!/usr/bin/env python3
"""Process every session listed in info.xml and print a full attendance matrix.

    python run_all.py --reset

``--reset`` deletes the database first so that the run is repeatable, which is
what the regression evidence in the report is based on.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from attendance.database import AttendanceDatabase
from attendance.image_processor import AttendanceImageProcessor
from attendance.models import ProcessingResult
from attendance.xml_loader import load_course_info


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Process every session listed in info.xml")
    parser.add_argument("--info", default="info.xml")
    parser.add_argument("--images", default="input_signing_sheets")
    parser.add_argument("--database", default="data/attendance.db")
    parser.add_argument("--output", default="output")
    parser.add_argument("--reset", action="store_true", help="Delete the existing database first")
    parser.add_argument("--quiet", action="store_true", help="Hide the per-stage progress lines")
    return parser


def main() -> int:
    arguments = build_parser().parse_args()

    try:
        database_path = Path(arguments.database)
        if arguments.reset and database_path.exists():
            database_path.unlink()

        course = load_course_info(arguments.info)
        processor = AttendanceImageProcessor(course)
        results: list[ProcessingResult] = []

        with AttendanceDatabase(database_path) as database:
            for session in course.sessions:
                image_path = Path(arguments.images) / session.image
                print(f"\n=== Processing {image_path.name} ({session.date}) ===")
                result = processor.process(
                    image_path,
                    session.date,
                    output_root=arguments.output,
                    progress=None if arguments.quiet else lambda m: print(f"[PROCESS] {m}"),
                )
                database.save_result(course, result)
                results.append(result)

        _print_matrix(course, results)
        print(f"\nDatabase: {database_path}")
        print(f"Attendance records: {sum(len(result.rows) for result in results)}")
        return 0
    except Exception as error:  # noqa: BLE001 - CLI boundary
        print(f"ERROR: {error}", file=sys.stderr)
        return 1


def _print_matrix(course, results: list[ProcessingResult]) -> None:
    """Print the present/absent matrix used as testing evidence in the report."""
    header = f"{'Session':<24}" + "".join(f"{s.index:>11}" for s in course.students) + f"{'Present':>10}"
    print("\nAttendance matrix (P = present, A = absent)")
    print("-" * len(header))
    print(header)
    print("-" * len(header))

    total_cells = 0
    total_present = 0
    for result in results:
        marks = "".join(f"{'P' if row.present else 'A':>11}" for row in result.rows)
        label = f"{result.image_path.name} {result.session_date}"
        print(f"{label:<24}{marks}{result.present_count:>7}/{len(result.rows)}")
        total_cells += len(result.rows)
        total_present += result.present_count
    print("-" * len(header))

    per_student = [
        sum(result.rows[position].present for result in results)
        for position in range(len(course.students))
    ]
    print(
        f"{'Present per student':<24}"
        + "".join(f"{value:>11}" for value in per_student)
        + f"{total_present:>7}/{total_cells}"
    )
    rate = 100.0 * total_present / total_cells if total_cells else 0.0
    grid_ok = sum(result.grid_detected for result in results)
    annotated = [
        (result.session_date, row.student.index)
        for result in results
        for row in result.rows
        if row.reason == "absence annotation"
    ]
    print(f"\nOverall attendance rate: {rate:.1f}%")
    print(f"Sheets whose printed grid was measured directly: {grid_ok}/{len(results)}")
    if annotated:
        print("Absences recorded by a written mark rather than an empty cell:")
        for date, index in annotated:
            print(f"    {index} on {date}")


if __name__ == "__main__":
    raise SystemExit(main())
