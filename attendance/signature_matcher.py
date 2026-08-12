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

# --------------------------------------------------------------- preparation


def _fallback_ssim(first: np.ndarray, second: np.ndarray) -> float:
    """SSIM following Wang et al. (2004), used when scikit-image is absent."""
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    a = first.astype(np.float64)
    b = second.astype(np.float64)
    kernel = (11, 11)
    sigma = 1.5
    mu_a = cv2.GaussianBlur(a, kernel, sigma)
    mu_b = cv2.GaussianBlur(b, kernel, sigma)
    mu_a2, mu_b2, mu_ab = mu_a * mu_a, mu_b * mu_b, mu_a * mu_b
    sigma_a2 = cv2.GaussianBlur(a * a, kernel, sigma) - mu_a2
    sigma_b2 = cv2.GaussianBlur(b * b, kernel, sigma) - mu_b2
    sigma_ab = cv2.GaussianBlur(a * b, kernel, sigma) - mu_ab
    ssim_map = ((2 * mu_ab + c1) * (2 * sigma_ab + c2)) / (
        (mu_a2 + mu_b2 + c1) * (sigma_a2 + sigma_b2 + c2)
    )
    return float(ssim_map.mean())


def _ssim(first: np.ndarray, second: np.ndarray) -> float:
    if HAS_SCIKIT_IMAGE:
        return float(_skimage_ssim(first, second, data_range=255))
    return _fallback_ssim(first, second)


def normalize_signature(image_path: str | Path, size: tuple[int, int] = CANVAS) -> np.ndarray:
    """Crop a signature mask to its ink, rescale it and centre it on a fixed canvas.

    Normalisation removes the differences in cell position and signature size
    that would otherwise dominate every similarity measure.
    """
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Signature image not found: {image_path}")
    _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Discard specks so that a single noise pixel cannot define the crop box.
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary)
    if count > 1:
        largest = max(stats[1:, cv2.CC_STAT_AREA])
        keep = np.zeros_like(binary)
        for component in range(1, count):
            if stats[component, cv2.CC_STAT_AREA] >= max(4, 0.02 * largest):
                keep[labels == component] = 255
        binary = keep

    canvas = np.zeros((size[1], size[0]), dtype=np.uint8)
    points = cv2.findNonZero(binary)
    if points is None:
        return canvas

    x, y, width, height = cv2.boundingRect(points)
    cropped = binary[y : y + height, x : x + width]
    available_width = size[0] - 24
    available_height = size[1] - 24
    scale = min(available_width / max(width, 1), available_height / max(height, 1))
    resized = cv2.resize(
        cropped,
        (max(1, int(width * scale)), max(1, int(height * scale))),
        interpolation=cv2.INTER_AREA,
    )
    start_x = (size[0] - resized.shape[1]) // 2
    start_y = (size[1] - resized.shape[0]) // 2
    canvas[start_y : start_y + resized.shape[0], start_x : start_x + resized.shape[1]] = resized
    return canvas


def _soften(mask: np.ndarray) -> np.ndarray:
    """Thicken and blur strokes so small pen-path differences do not zero the overlap.

    Two genuine signatures almost never place identical pixels on top of each
    other. Comparing hard one-pixel strokes therefore measures registration
    error, not similarity.
    """
    thick = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    return cv2.GaussianBlur(thick, (11, 11), 3.5)


# ------------------------------------------------------------------ measures


def _ncc_similarity(first: np.ndarray, second: np.ndarray) -> float:
    """Normalised cross-correlation that tolerates a small residual translation.

    The centre of one image is slid over the other and the best correlation is
    kept, so a signature written a few pixels higher in the cell is not
    penalised for its position. Both directions are evaluated so that the
    measure is symmetric.
    """
    if not first.any() or not second.any():
        return 0.0

    def best_response(haystack: np.ndarray, needle: np.ndarray) -> float:
        inner = needle[SHIFT_TOLERANCE:-SHIFT_TOLERANCE, SHIFT_TOLERANCE:-SHIFT_TOLERANCE]
        if inner.size == 0:
            return 0.0
        response = cv2.matchTemplate(
            haystack.astype(np.float32), inner.astype(np.float32), cv2.TM_CCOEFF_NORMED
        )
        return float(response.max())

    return max(0.0, best_response(first, second), best_response(second, first))


def _iou_similarity(first: np.ndarray, second: np.ndarray) -> float:
    """Intersection over union of the two softened ink areas."""
    first_ink = first > 10
    second_ink = second > 10
    union = int(np.logical_or(first_ink, second_ink).sum())
    if union == 0:
        return 0.0
    return float(np.logical_and(first_ink, second_ink).sum() / union)


def _profile_similarity(first: np.ndarray, second: np.ndarray) -> float:
    """Correlate the horizontal and vertical ink projection profiles."""
    if not first.any() or not second.any():
        return 0.0

    def correlate(a: np.ndarray, b: np.ndarray) -> float:
        a = a.astype(np.float64)
        b = b.astype(np.float64)
        a -= a.mean()
        b -= b.mean()
        denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
        return float(np.dot(a, b) / denominator) if denominator else 0.0

    vertical = correlate(first.sum(axis=0), second.sum(axis=0))
    horizontal = correlate(first.sum(axis=1), second.sum(axis=1))
    return float(np.clip((vertical + horizontal) / 2.0, 0.0, 1.0))


def _orb_similarity(first: np.ndarray, second: np.ndarray) -> float:
    """Fraction of ORB descriptors that find a close mutual match."""
    orb = cv2.ORB_create(nfeatures=500, fastThreshold=5)
    _, first_descriptors = orb.detectAndCompute(first, None)
    _, second_descriptors = orb.detectAndCompute(second, None)
    if first_descriptors is None or second_descriptors is None:
        return 0.0
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(first_descriptors, second_descriptors)
    if not matches:
        return 0.0
    good_matches = [match for match in matches if match.distance <= 55]
    smaller_set = min(len(first_descriptors), len(second_descriptors))
    return min(1.0, len(good_matches) / max(8, smaller_set))



