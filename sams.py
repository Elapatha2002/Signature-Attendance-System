#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

from attendance.database import AttendanceDatabase
from attendance.image_processor import AttendanceImageProcessor
from attendance.xml_loader import date_for_image, load_course_info


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Process a signing-sheet image and store student attendance."
    )
    parser.add_argument("image", help="Path to the photographed signing sheet")
    parser.add_argument("info_xml", help="Path to info.xml")
    parser.add_argument("--date", help="Lecture date in DD/MM/YYYY format")
    parser.add_argument("--database", default="data/attendance.db")
    parser.add_argument("--output", default="output")
    return parser


def main() -> int:
    arguments = build_parser().parse_args()
    try:
        course = load_course_info(arguments.info_xml)
        session_date = arguments.date or date_for_image(course, arguments.image)
        if not session_date:
            raise ValueError(
                "The lecture date was not found in info.xml. Supply it using --date DD/MM/YYYY."
            )

        processor = AttendanceImageProcessor(course)
        result = processor.process(
            arguments.image,
            session_date,
            output_root=arguments.output,
            progress=lambda message: print(f"[PROCESS] {message}"),
        )
        with AttendanceDatabase(arguments.database) as database:
            database.save_result(course, result)

        print(f"\nAttendance result - {result.session_date}")
        print("-" * 112)
        print(
            f"{'Student Index':<16}{'Student Name':<38}{'Status':<9}"
            f"{'Colour':>9}{'Dark ink':>10}{'Conf.':>8}  Reason"
        )
        print("-" * 112)
        for row in result.rows:
            print(
                f"{row.student.index:<16}{row.student.name:<38}{row.status:<9}"
                f"{row.color_ratio:>9.4f}{row.residual_ratio:>10.4f}{row.confidence:>8.1%}"
                f"  {row.reason}"
            )
        print("-" * 112)
        print(
            f"Present: {result.present_count}/{len(result.rows)}   "
            f"Grid: {'measured from printed separators' if result.grid_detected else 'estimated (equal split fallback)'}"
        )
        print(f"Saved processing images to: {result.output_dir}")
        return 0
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
