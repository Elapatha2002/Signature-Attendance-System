#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

from attendance.database import AttendanceDatabase
from attendance.visualizer import create_student_chart


def main() -> int:
    parser = argparse.ArgumentParser(description="Display attendance for one student")
    parser.add_argument("student_index")
    parser.add_argument("--database", default="data/attendance.db")
    parser.add_argument("--output", help="Output PNG path")
    parser.add_argument("--show", action="store_true")
    arguments = parser.parse_args()

    output = arguments.output or f"output/charts/{arguments.student_index}_attendance.png"
    try:
        with AttendanceDatabase(arguments.database) as database:
            records = database.student_summary(arguments.student_index)
            if not records:
                raise ValueError(f"No records found for {arguments.student_index}")
            chart = create_student_chart(database, arguments.student_index, output, arguments.show)

        present = sum(record["status"] == "Present" for record in records)
        print(f"Student: {records[0]['name']} ({arguments.student_index})")
        for record in records:
            print(f"{record['session_date']}: {record['status']}")
        print(f"Attendance rate: {present / len(records) * 100:.1f}%")
        print(f"Chart saved to: {chart}")
        return 0
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
