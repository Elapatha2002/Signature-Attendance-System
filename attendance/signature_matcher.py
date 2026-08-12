"""Experimental offline signature comparison.

The extracted signature masks are low-resolution by-products of attendance
detection rather than controlled biometric captures, so this module produces a
*similarity warning* and never an identity decision. Five weak measures are
combined:

* shift-tolerant normalised cross-correlation (NCC) of the softened masks;
* intersection-over-union of the softened ink areas;
* correlation of the horizontal and vertical ink projection profiles;
* structural similarity (SSIM) of the softened masks;
* ORB binary-descriptor matching.

A sixth measure, Hu-moment shape distance, was implemented and then removed
from the score: measured on the coursework dataset it reached an area under the
ROC curve of only 0.56, which is close to random guessing, and including it
lowered the combined result.

The decision threshold is not a magic constant. :func:`calibrate_threshold`
derives it from the data by measuring the genuine (same student) and impostor
(different student) score distributions, which is what makes the reported
numbers interpretable.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from statistics import mean, pstdev

import cv2
import numpy as np

try:  # pragma: no cover - exercised only by the import path that is available
    from skimage.metrics import structural_similarity as _skimage_ssim

    HAS_SCIKIT_IMAGE = True
except ImportError:  # pragma: no cover
    _skimage_ssim = None
    HAS_SCIKIT_IMAGE = False


#: Weights of the five similarity measures. They sum to 1.0 and were chosen
#: from the measured single-measure ROC areas on the coursework dataset.
WEIGHTS = {"ncc": 0.35, "iou": 0.20, "profile": 0.25, "ssim": 0.10, "orb": 0.10}

CANVAS = (320, 160)

#: Pixels of translation tolerated by the cross-correlation search.
SHIFT_TOLERANCE = 18


@dataclass(frozen=True)
class SignatureComparison:
    reference: Path
    candidate: Path
    ncc_score: float
    iou_score: float
    profile_score: float
    ssim_score: float
    orb_score: float
    combined_score: float


@dataclass(frozen=True)
class Calibration:
    """Genuine/impostor score statistics and the threshold derived from them."""

    genuine_scores: list[float]
    impostor_scores: list[float]
    threshold: float

    @property
    def genuine_mean(self) -> float:
        return mean(self.genuine_scores) if self.genuine_scores else 0.0

    @property
    def impostor_mean(self) -> float:
        return mean(self.impostor_scores) if self.impostor_scores else 0.0

    @property
    def separation(self) -> float:
        """Fisher-style separation between the two distributions."""
        spread = pstdev(self.genuine_scores or [0.0]) + pstdev(self.impostor_scores or [0.0])
        return (self.genuine_mean - self.impostor_mean) / spread if spread else 0.0

    @property
    def roc_auc(self) -> float:
        """Probability that a genuine pair outscores an impostor pair.

        0.5 means the measure is no better than guessing and 1.0 means the two
        distributions are perfectly separated.
        """
        if not self.genuine_scores or not self.impostor_scores:
            return 0.5
        wins = sum(
            1.0 if genuine > impostor else 0.5 if genuine == impostor else 0.0
            for genuine in self.genuine_scores
            for impostor in self.impostor_scores
        )
        return wins / (len(self.genuine_scores) * len(self.impostor_scores))

    @property
    def genuine_accepted(self) -> int:
        return sum(score >= self.threshold for score in self.genuine_scores)

    @property
    def impostor_rejected(self) -> int:
        return sum(score < self.threshold for score in self.impostor_scores)


