import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from rtdetr_eval.evaluate import compute_iou, run_evaluation


class TestEvaluate(unittest.TestCase):
    def test_compute_iou_basic(self):
        a = [0, 0, 10, 10]
        b = [5, 5, 15, 15]
        iou = compute_iou(a, b)
        # Overlap area = 25, union = 100 + 100 - 25 = 175
        self.assertAlmostEqual(iou, 25 / 175, places=6)

    def test_run_evaluation_handles_null_confidence(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            gt_csv = td / "gt.csv"
            pred_csv = td / "pred.csv"
            out_dir = td / "out"

            # One GT box for class 1 on frame 0
            gt_rows = [
                {
                    "frame": 0,
                    "predictions_json": json.dumps(
                        [{"bbox": [0, 0, 10, 10], "class_id": 1, "confidence": 1.0}]
                    ),
                }
            ]

            # Two predictions: one has null confidence (should not crash sorting),
            # and one matching TP with confidence 0.9.
            pred_rows = [
                {
                    "frame": 0,
                    "predictions_json": json.dumps(
                        [
                            {"bbox": [0, 0, 10, 10], "class_id": 1, "confidence": 0.9},
                            {"bbox": [50, 50, 60, 60], "class_id": 1, "confidence": None},
                        ]
                    ),
                }
            ]

            pd.DataFrame(gt_rows).to_csv(gt_csv, index=False)
            pd.DataFrame(pred_rows).to_csv(pred_csv, index=False)

            metrics = run_evaluation(
                gt_csv=gt_csv,
                pred_csv=pred_csv,
                out_dir=out_dir,
                iou_thresh=0.5,
                plots=False,
                write_json=False,
            )

            self.assertIn("classes", metrics)
            self.assertIn("Pedestrian", metrics["classes"])
            self.assertGreaterEqual(metrics["classes"]["Pedestrian"]["ap"], 0.0)

    def test_run_evaluation_handles_weird_confidence_values(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            gt_csv = td / "gt.csv"
            pred_csv = td / "pred.csv"
            out_dir = td / "out"

            gt_rows = [
                {
                    "frame": 0,
                    "predictions_json": json.dumps(
                        [{"bbox": [0, 0, 10, 10], "class_id": 1, "confidence": 1.0}]
                    ),
                }
            ]

            # confidence can be missing, a string, or an invalid type; evaluation should not crash.
            pred_rows = [
                {
                    "frame": 0,
                    "predictions_json": json.dumps(
                        [
                            {"bbox": [0, 0, 10, 10], "class_id": 1, "confidence": "0.9"},
                            {"bbox": [0, 0, 10, 10], "class_id": 1},  # missing
                            {"bbox": [50, 50, 60, 60], "class_id": 1, "confidence": {"bad": "type"}},
                        ]
                    ),
                }
            ]

            pd.DataFrame(gt_rows).to_csv(gt_csv, index=False)
            pd.DataFrame(pred_rows).to_csv(pred_csv, index=False)

            metrics = run_evaluation(
                gt_csv=gt_csv,
                pred_csv=pred_csv,
                out_dir=out_dir,
                iou_thresh=0.5,
                plots=False,
                write_json=False,
            )

            self.assertIn("classes", metrics)
            self.assertIn("Pedestrian", metrics["classes"])

