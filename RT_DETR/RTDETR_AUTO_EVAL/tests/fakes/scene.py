"""
Synthetic scene + analytic objective for the e2e test.

Pure stdlib (no rtdetr_eval / numpy import) so it can be imported both by the
test process and by the fake_infer.py subprocess. The detector "quality" is a
multi-term polynomial response surface in three swept hyperparameters with a
known interior optimum, so the search has a real hill to climb and the
leaderboard is deterministic for a fixed search seed.
"""

from __future__ import annotations

import json
from pathlib import Path

# (low, high) bounds, mirrored from rtdetr_eval.search.SEARCH_SPACE but hardcoded
# to keep this module dependency-free in a subprocess.
BOUNDS = {
    "inference.confidence": (0.40, 0.85),
    "nms.hard_iou": (0.15, 0.40),
    "classes.1.min_confidence": (0.45, 0.75),
}

# Interior optimum (raw param values) where quality() peaks.
OPTIMUM = {
    "inference.confidence": 0.60,
    "nms.hard_iou": 0.25,
    "classes.1.min_confidence": 0.60,
}

FP_MAX = 100  # max false positives per frame at q=0; large so precision (hence AP)
              # varies *continuously* with q instead of snapping to 11-point steps
N_FRAMES = 20  # frames in the synthetic clip (shared by GT writer and fake_infer)
N_PED = 40    # pedestrians (class 1) per frame -> fine-grained, smooth recall
N_CYC = 24    # cyclists  (class 0) per frame


def _normalize(path: str, value: float) -> float:
    low, high = BOUNDS[path]
    return (float(value) - low) / (high - low)


def quality(params: dict) -> float:
    """Polynomial response surface in [0.05, 0.98] with a peak at OPTIMUM.

    Concave (quadratic) bowls around the optimum for each param, plus a small
    cross term and a cubic term so it is not a trivial single-degree surface.
    """
    xc = _normalize("inference.confidence", params.get("inference.confidence", 0.60))
    xh = _normalize("nms.hard_iou", params.get("nms.hard_iou", 0.25))
    xp = _normalize("classes.1.min_confidence", params.get("classes.1.min_confidence", 0.60))

    oc = _normalize("inference.confidence", OPTIMUM["inference.confidence"])
    oh = _normalize("nms.hard_iou", OPTIMUM["nms.hard_iou"])
    op = _normalize("classes.1.min_confidence", OPTIMUM["classes.1.min_confidence"])

    dc, dh, dp = xc - oc, xh - oh, xp - op

    # Steep surface so high quality is rare and the search has real work to do.
    q = 1.0
    q -= 1.70 * dc * dc      # quadratic bowl (confidence)
    q -= 1.10 * dh * dh      # quadratic bowl (hard_iou)
    q -= 1.40 * dp * dp      # quadratic bowl (ped min_confidence)
    q -= 0.60 * dc * dp      # cross term
    q -= 0.45 * dc * dc * dc  # cubic asymmetry

    return max(0.02, min(0.99, q))


def param_distance(params: dict) -> float:
    """Normalized L2 distance from a config to the optimum (for convergence checks)."""
    total = 0.0
    for path, opt in OPTIMUM.items():
        if path in params:
            total += (_normalize(path, params[path]) - _normalize(path, opt)) ** 2
    return total ** 0.5


# ---------------------------------------------------------------------------
# Ground-truth scene
# ---------------------------------------------------------------------------

def gt_for_frame(frame: int) -> list[dict]:
    """Deterministic GT boxes for a frame: N_PED pedestrians + N_CYC cyclists, well spaced."""
    boxes = []
    for i in range(N_PED):  # pedestrians (class 1)
        x = 40 + 70 * i + (frame % 5) * 3
        y = 40 + (frame % 7) * 2
        boxes.append({"class_id": 1, "bbox": [x, y, x + 40, y + 90], "confidence": 1.0})
    for i in range(N_CYC):  # cyclists (class 0)
        x = 60 + 110 * i + (frame % 4) * 4
        y = 400 + (frame % 6) * 2
        boxes.append({"class_id": 0, "bbox": [x, y, x + 70, y + 70], "confidence": 1.0})
    return boxes


def write_gt_csv(path: Path, n_frames: int) -> None:
    rows = ["frame,predictions_json"]
    for f in range(n_frames):
        payload = json.dumps(gt_for_frame(f)).replace('"', '""')
        rows.append(f'{f},"{payload}"')
    Path(path).write_text("\n".join(rows) + "\n")
