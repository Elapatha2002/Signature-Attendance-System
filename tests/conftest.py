from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from attendance.image_processor import AttendanceImageProcessor  # noqa: E402
from attendance.xml_loader import load_course_info  # noqa: E402


@pytest.fixture(scope="session")
def project_root() -> Path:
    return ROOT


@pytest.fixture(scope="session")
def course(project_root: Path):
    return load_course_info(project_root / "info.xml")


@pytest.fixture(scope="session")
def results(project_root: Path, course, tmp_path_factory):
    """Process all five supplied sheets once and share the results between tests."""
    output_root = tmp_path_factory.mktemp("processing")
    processor = AttendanceImageProcessor(course)
    return {
        session.image: processor.process(
            project_root / "input_signing_sheets" / session.image,
            session.date,
            output_root=output_root,
        )
        for session in course.sessions
    }
