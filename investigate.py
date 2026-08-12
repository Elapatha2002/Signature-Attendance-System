#!/usr/bin/env python3
"""Experimental signature-consistency check for one student.

    python investigate.py 10000409

The program compares the signatures the attendance system already extracted for
the given student. Because a similarity number is meaningless without a
baseline, it first calibrates the decision threshold on the whole dataset by
measuring genuine (same student) and impostor (different student) score
distributions, then reports where this student's own comparisons fall.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg", force=False)
import matplotlib.pyplot as plt  # noqa: E402

from attendance.database import AttendanceDatabase  # noqa: E402
from attendance.signature_matcher import (  # noqa: E402
    HAS_SCIKIT_IMAGE,
    calibrate_threshold,
    collect_samples,
    compare_signatures,
)

REVIEW_COLOUR = "#C62828"
SIMILAR_COLOUR = "#2E7D32"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare a student's detected signatures across lecture dates"
    )
    parser.add_argument("student_index")
    parser.add_argument("--database", default="data/attendance.db")
    parser.add_argument("--output", help="Output comparison chart")
    parser.add_argument(
        "--threshold",
        type=float,
        help="Override the automatically calibrated similarity threshold",
    )
    return parser


def main() -> int:
    arguments = build_parser().parse_args()
    try:
        with AttendanceDatabase(arguments.database) as database:
            samples = collect_samples(database.all_signature_paths())
            student_records = [
                record
                for record in database.student_summary(arguments.student_index)
                if record["status"] == "Present" and record["signature_path"]
            ]
            student_name = student_records[0]["name"] if student_records else ""

        # Pair dates with files from a single date-ordered source so that a
        # score can never be reported against the wrong lecture.
        own_samples = [
            (record["session_date"], Path(record["signature_path"]))
            for record in student_records
            if Path(record["signature_path"]) in set(samples.get(arguments.student_index, []))
        ]
        if len(own_samples) < 2:
            raise ValueError(
                f"At least two extracted signatures are required for {arguments.student_index}. "
                "Run run_all.py first."
            )

        calibration = calibrate_threshold(samples)
        threshold = (
            arguments.threshold if arguments.threshold is not None else calibration.threshold
        )

        print(f"Structural similarity backend: {'scikit-image' if HAS_SCIKIT_IMAGE else 'built-in'}")
        print("\nThreshold calibration on the whole dataset")
        print("-" * 78)
        print(
            f"Genuine pairs (same student)      : {len(calibration.genuine_scores):>3}  "
            f"mean {calibration.genuine_mean:.3f}"
        )
        print(
            f"Impostor pairs (different student): {len(calibration.impostor_scores):>3}  "
            f"mean {calibration.impostor_mean:.3f}"
        )
        print(f"ROC area under curve             : {calibration.roc_auc:.3f}")
        print(f"Calibrated threshold             : {threshold:.3f}")
        print(
            f"Genuine pairs accepted           : {calibration.genuine_accepted}"
            f"/{len(calibration.genuine_scores)}"
        )
        print(
            f"Impostor pairs rejected          : {calibration.impostor_rejected}"
            f"/{len(calibration.impostor_scores)}"
        )

        reference_date, reference_path = own_samples[0]
        comparisons = [
            (date, compare_signatures(reference_path, path))
            for date, path in own_samples[1:]
        ]

        print(f"\nReference signature: {reference_date} - {reference_path}")
        print(
            f"{'Date':<12}{'NCC':<8}{'IoU':<8}{'Profile':<9}{'SSIM':<8}"
            f"{'ORB':<8}{'Combined':<10}Decision"
        )
        print("-" * 78)
        for date, comparison in comparisons:
            decision = "Consistent" if comparison.combined_score >= threshold else "Review"
            print(
                f"{date:<12}{comparison.ncc_score:<8.3f}{comparison.iou_score:<8.3f}"
                f"{comparison.profile_score:<9.3f}{comparison.ssim_score:<8.3f}"
                f"{comparison.orb_score:<8.3f}{comparison.combined_score:<10.3f}{decision}"
            )

        output = Path(
            arguments.output
            or f"output/signature_comparisons/{arguments.student_index}_comparison.png"
        )
        _plot(arguments.student_index, student_name, comparisons, calibration, threshold, output)
        print(f"\nComparison chart saved to: {output}")
        print(
            "Note: with an ROC area of "
            f"{calibration.roc_auc:.2f} this measure is a weak similarity warning only. "
            "It prioritises manual review and is not proof of identity or forgery."
        )
        return 0
    except Exception as error:  # noqa: BLE001 - CLI boundary
        print(f"ERROR: {error}", file=sys.stderr)
        return 1


def _plot(student_index, student_name, comparisons, calibration, threshold, output: Path) -> None:
    figure, (score_axis, distribution_axis) = plt.subplots(
        1, 2, figsize=(13, 5.4), gridspec_kw={"width_ratios": [3, 2]}
    )

    dates = [date for date, _ in comparisons]
    scores = [comparison.combined_score for _, comparison in comparisons]
    bars = score_axis.bar(
        dates,
        scores,
        color=[SIMILAR_COLOUR if score >= threshold else REVIEW_COLOUR for score in scores],
        width=0.55,
    )
    score_axis.axhline(
        threshold,
        linestyle="--",
        color="#37474F",
        linewidth=1.4,
        label=f"Calibrated threshold {threshold:.2f}",
    )
    score_axis.set_ylim(0, 1)
    score_axis.set_xlabel("Lecture date compared with the reference signature")
    score_axis.set_ylabel("Combined similarity score")
    score_axis.set_title("Signature consistency per lecture", fontsize=11)
    score_axis.bar_label(bars, fmt="%.2f", padding=3)
    score_axis.legend(loc="upper right")
    score_axis.grid(axis="y", alpha=0.3)

    distribution_axis.hist(
        calibration.impostor_scores,
        bins=18,
        alpha=0.65,
        color=REVIEW_COLOUR,
        label=f"Impostor pairs (n={len(calibration.impostor_scores)})",
        density=True,
    )
    distribution_axis.hist(
        calibration.genuine_scores,
        bins=18,
        alpha=0.65,
        color=SIMILAR_COLOUR,
        label=f"Genuine pairs (n={len(calibration.genuine_scores)})",
        density=True,
    )
    distribution_axis.axvline(threshold, linestyle="--", color="#37474F", linewidth=1.4)
    distribution_axis.set_xlabel("Combined similarity score")
    distribution_axis.set_ylabel("Density")
    distribution_axis.set_title(
        f"Score distributions (ROC AUC {calibration.roc_auc:.2f})", fontsize=11
    )
    distribution_axis.legend(fontsize=8)
    distribution_axis.grid(axis="y", alpha=0.3)

    figure.suptitle(
        f"Signature Investigation - {student_name} ({student_index})",
        fontsize=13,
        fontweight="bold",
    )
    figure.tight_layout(rect=(0, 0, 1, 0.93))
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)


if __name__ == "__main__":
    raise SystemExit(main())
