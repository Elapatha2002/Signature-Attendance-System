from __future__ import annotations

from pathlib import Path
import sqlite3
from typing import Iterable

from .models import CourseInfo, ProcessingResult


class AttendanceDatabase:
    """Small SQLite repository used by the command-line programs."""

    def __init__(self, path: str | Path = "data/attendance.db") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.path)
        self.connection.row_factory = sqlite3.Row
        self._create_schema()

    def _create_schema(self) -> None:
        self.connection.executescript(
            """
            PRAGMA foreign_keys = ON;

            CREATE TABLE IF NOT EXISTS students (
                student_index TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                name TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS subjects (
                subject_code TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                lecturer TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                subject_code TEXT NOT NULL,
                session_date TEXT NOT NULL,
                image_file TEXT NOT NULL,
                UNIQUE(subject_code, session_date),
                FOREIGN KEY(subject_code) REFERENCES subjects(subject_code)
            );

            CREATE TABLE IF NOT EXISTS attendance (
                session_id INTEGER NOT NULL,
                student_index TEXT NOT NULL,
                status TEXT NOT NULL CHECK(status IN ('Present', 'Absent')),
                confidence REAL NOT NULL,
                color_pixels INTEGER NOT NULL,
                residual_pixels INTEGER NOT NULL,
                signature_path TEXT,
                PRIMARY KEY(session_id, student_index),
                FOREIGN KEY(session_id) REFERENCES sessions(id) ON DELETE CASCADE,
                FOREIGN KEY(student_index) REFERENCES students(student_index)
            );
            """
        )
        self.connection.commit()

    def seed(self, course: CourseInfo) -> None:
        self.connection.execute(
            """INSERT INTO subjects(subject_code, title, lecturer)
               VALUES (?, ?, ?)
               ON CONFLICT(subject_code) DO UPDATE SET
                   title=excluded.title, lecturer=excluded.lecturer""",
            (course.subject.code, course.subject.title, course.subject.lecturer),
        )
        self.connection.executemany(
            """INSERT INTO students(student_index, title, name)
               VALUES (?, ?, ?)
               ON CONFLICT(student_index) DO UPDATE SET
                   title=excluded.title, name=excluded.name""",
            [(student.index, student.title, student.name) for student in course.students],
        )
        self.connection.commit()

    def save_result(self, course: CourseInfo, result: ProcessingResult) -> None:
        self.seed(course)
        self.connection.execute(
            """INSERT INTO sessions(subject_code, session_date, image_file)
               VALUES (?, ?, ?)
               ON CONFLICT(subject_code, session_date) DO UPDATE SET
                   image_file=excluded.image_file""",
            (course.subject.code, result.session_date, result.image_path.name),
        )
        # sqlite3_last_insert_rowid() is not updated when an upsert takes the
        # DO UPDATE branch, so it can still hold the rowid of an unrelated
        # INSERT made by seed(). The session id is therefore always read back.
        session_id = self.connection.execute(
            "SELECT id FROM sessions WHERE subject_code=? AND session_date=?",
            (course.subject.code, result.session_date),
        ).fetchone()["id"]

        self.connection.executemany(
            """INSERT INTO attendance(
                    session_id, student_index, status, confidence,
                    color_pixels, residual_pixels, signature_path
               ) VALUES (?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(session_id, student_index) DO UPDATE SET
                    status=excluded.status,
                    confidence=excluded.confidence,
                    color_pixels=excluded.color_pixels,
                    residual_pixels=excluded.residual_pixels,
                    signature_path=excluded.signature_path""",
            [
                (
                    session_id,
                    row.student.index,
                    "Present" if row.present else "Absent",
                    row.confidence,
                    row.color_pixels,
                    row.residual_pixels,
                    str(row.signature_path) if row.signature_path else None,
                )
                for row in result.rows
            ],
        )
        self.connection.commit()

    def student_summary(self, student_index: str) -> list[sqlite3.Row]:
        return self.connection.execute(
            """SELECT s.session_date, a.status, a.confidence, a.signature_path,
                      st.name, st.title, s.subject_code
               FROM attendance a
               JOIN sessions s ON s.id=a.session_id
               JOIN students st ON st.student_index=a.student_index
               WHERE a.student_index=?
               ORDER BY date(substr(s.session_date,7,4)||'-'||substr(s.session_date,4,2)||'-'||substr(s.session_date,1,2))""",
            (student_index,),
        ).fetchall()

    def all_signature_paths(self) -> list[sqlite3.Row]:
        """Return every stored signature file with its student and session."""
        return self.connection.execute(
            """SELECT a.student_index, a.status, a.signature_path, s.session_date
               FROM attendance a
               JOIN sessions s ON s.id = a.session_id
               WHERE a.signature_path IS NOT NULL
               ORDER BY a.student_index,
                        date(substr(s.session_date,7,4)||'-'||substr(s.session_date,4,2)||'-'||substr(s.session_date,1,2))"""
        ).fetchall()

    def class_summary(self) -> list[sqlite3.Row]:
        """Return one row per student with present, absent and rate columns."""
        return self.connection.execute(
            """SELECT st.student_index, st.name,
                      SUM(a.status = 'Present') AS present_count,
                      SUM(a.status = 'Absent')  AS absent_count,
                      COUNT(*)                  AS session_count,
                      ROUND(100.0 * SUM(a.status = 'Present') / COUNT(*), 1) AS rate
               FROM attendance a
               JOIN students st ON st.student_index = a.student_index
               GROUP BY st.student_index, st.name
               ORDER BY st.student_index"""
        ).fetchall()

    def session_summary(self) -> list[sqlite3.Row]:
        """Return one row per lecture session with the present/absent totals."""
        return self.connection.execute(
            """SELECT s.session_date, s.image_file,
                      SUM(a.status = 'Present') AS present_count,
                      COUNT(*)                  AS student_count
               FROM attendance a
               JOIN sessions s ON s.id = a.session_id
               GROUP BY s.id
               ORDER BY date(substr(s.session_date,7,4)||'-'||substr(s.session_date,4,2)||'-'||substr(s.session_date,1,2))"""
        ).fetchall()

    def all_results(self) -> list[sqlite3.Row]:
        return self.connection.execute(
            """SELECT s.session_date, st.student_index, st.name, a.status,
                      a.confidence, a.color_pixels, a.residual_pixels
               FROM attendance a
               JOIN sessions s ON s.id=a.session_id
               JOIN students st ON st.student_index=a.student_index
               ORDER BY date(substr(s.session_date,7,4)||'-'||substr(s.session_date,4,2)||'-'||substr(s.session_date,1,2)),
                        st.student_index"""
        ).fetchall()

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> "AttendanceDatabase":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()