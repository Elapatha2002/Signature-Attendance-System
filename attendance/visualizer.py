"""Matplotlib attendance visualizations.

Two charts are produced:

* :func:`create_student_chart` - the per-student history required by
  ``infovis.py``;
* :func:`create_class_chart` - a class-wide comparison used in the report.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=False)
import matplotlib.pyplot as plt  # noqa: E402  (backend must be selected first)

from .database import AttendanceDatabase  # noqa: E402

PRESENT_COLOUR = "#2E7D32"
ABSENT_COLOUR = "#C62828"
GRID_COLOUR = "#B0BEC5"


def _style(axis) -> None:
    axis.set_axisbelow(True)
    axis.grid(axis="y", color=GRID_COLOUR, alpha=0.35, linewidth=0.8)
    for edge in ("top", "right"):
        axis.spines[edge].set_visible(False)


def create_student_chart(
    database: AttendanceDatabase,
    student_index: str,
    output_path: str | Path,
    show: bool = False,
) -> Path:
    """Plot one student's attendance history as a labelled categorical bar chart.

    Absent sessions are drawn as full-height red bars rather than zero-height
    bars, so that a missed lecture is as visible as an attended one.
    """
    records = database.student_summary(student_index)
    if not records:
        raise ValueError(
            f"No attendance records exist for student index {student_index}. Run sams.py first."
        )

    dates = [record["session_date"] for record in records]
    present_flags = [record["status"] == "Present" for record in records]
    confidences = [float(record["confidence"]) for record in records]
    present_count = sum(present_flags)
    total_count = len(present_flags)
    rate = present_count / total_count * 100

    figure, (status_axis, rate_axis) = plt.subplots(
        1, 2, figsize=(12, 5.2), gridspec_kw={"width_ratios": [3, 1]}
    )

    bars = status_axis.bar(
        dates,
        [1] * total_count,
        color=[PRESENT_COLOUR if flag else ABSENT_COLOUR for flag in present_flags],
        edgecolor="white",
        linewidth=1.2,
        width=0.62,
    )
    for bar, flag, confidence in zip(bars, present_flags, confidences):
        centre = bar.get_x() + bar.get_width() / 2
        status_axis.text(
            centre,
            0.54,
            "PRESENT" if flag else "ABSENT",
            ha="center",
            va="center",
            color="white",
            fontsize=9,
            fontweight="bold",
        )
        status_axis.text(
            centre,
            0.46,
            f"{confidence:.0%} conf.",
            ha="center",
            va="center",
            color="white",
            fontsize=8,
        )
    status_axis.set_ylim(0, 1)
    status_axis.set_yticks([])
    status_axis.set_xlabel("Lecture date")
    status_axis.set_title("Attendance per lecture", fontsize=11)
    for edge in ("top", "right", "left"):
        status_axis.spines[edge].set_visible(False)

    rate_axis.bar(
        ["Present", "Absent"],
        [present_count, total_count - present_count],
        color=[PRESENT_COLOUR, ABSENT_COLOUR],
        width=0.55,
    )
    rate_axis.set_ylim(0, max(total_count, 1) * 1.2)
    rate_axis.set_ylabel("Number of lectures")
    rate_axis.set_title("Totals", fontsize=11)
    for index, value in enumerate([present_count, total_count - present_count]):
        rate_axis.text(index, value + 0.08, str(value), ha="center", fontweight="bold")
    _style(rate_axis)

    figure.suptitle(
        f"Attendance Summary - {records[0]['name']} ({student_index})\n"
        f"Present {present_count}/{total_count} lectures | Attendance rate {rate:.1f}%",
        fontsize=13,
        fontweight="bold",
    )
    figure.tight_layout(rect=(0, 0, 1, 0.92))

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180)
    if show:
        plt.show()
    plt.close(figure)
    return output_path


def create_class_chart(
    database: AttendanceDatabase,
    output_path: str | Path,
    show: bool = False,
) -> Path:
    """Plot the attendance rate of every student in the subject."""
    records = database.class_summary()
    if not records:
        raise ValueError("The database contains no attendance records. Run run_all.py first.")

    labels = [f"{record['student_index']}\n{_short_name(record['name'])}" for record in records]
    rates = [float(record["rate"]) for record in records]
    present = [int(record["present_count"]) for record in records]
    totals = [int(record["session_count"]) for record in records]
    average = sum(rates) / len(rates)

    figure, axis = plt.subplots(figsize=(11, 5.6))
    bars = axis.bar(
        labels,
        rates,
        color=[PRESENT_COLOUR if rate >= average else ABSENT_COLOUR for rate in rates],
        width=0.6,
    )
    axis.axhline(
        average,
        linestyle="--",
        linewidth=1.4,
        color="#37474F",
        label=f"Class average {average:.1f}%",
    )
    axis.set_ylim(0, 118)
    axis.set_ylabel("Attendance rate (%)")
    axis.set_xlabel("Student")
    axis.set_title(
        f"Attendance Rate per Student ({totals[0]} lectures)", fontsize=13, fontweight="bold"
    )
    # Labels sit inside the bar so that they never collide with the average line.
    for bar, rate, count, total in zip(bars, rates, present, totals):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            rate - 6,
            f"{rate:.0f}%\n({count}/{total})",
            ha="center",
            va="top",
            fontsize=9,
            color="white",
            fontweight="bold",
        )
    axis.legend(loc="upper center", bbox_to_anchor=(0.5, 1.0), frameon=False, fontsize=9)
    axis.tick_params(axis="x", labelsize=8)
    _style(axis)
    figure.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180)
    if show:
        plt.show()
    plt.close(figure)
    return output_path


def _short_name(name: str, limit: int = 22) -> str:
    return name if len(name) <= limit else name[: limit - 1] + "."
