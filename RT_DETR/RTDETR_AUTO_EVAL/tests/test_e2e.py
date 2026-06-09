"""
End-to-end test of the two-phase warm-start search.

Uses a stub inference script (tests/fakes/fake_infer.py) whose detection quality
is a polynomial of the hyperparameters, scored by the REAL run_evaluation. No
torch / cv2 / ultralytics and no real video frames are involved.
"""

import json
import shutil
import tempfile
import unittest
from pathlib import Path

from rtdetr_eval.search import run_exploitation, run_exploration
from rtdetr_eval.trials import run_trials
from tests.fakes import scene

FAKES = Path(__file__).resolve().parent / "fakes"
BASE_YAML = FAKES / "base.yaml"
FAKE_INFER = FAKES / "fake_infer.py"

SEED = 50
N_EXPLORE = 12
N_EXPLOIT = 9
TOP_K = 3


def _mean_distance(manifest: dict, prefix: str) -> float:
    dists = [scene.param_distance(p) for name, p in manifest.items() if name.startswith(prefix)]
    return sum(dists) / len(dists) if dists else float("inf")


class TestE2ETwoPhase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.work = Path(self._tmp.name)
        self.base = self.work / "base.yaml"
        shutil.copyfile(BASE_YAML, self.base)
        self.video = self.work / "clip.mp4"
        self.video.touch()
        self.gt = self.work / "ground_truth.csv"
        scene.write_gt_csv(self.gt, scene.N_FRAMES)
        self.trials_dir = self.work / "trials"

    def tearDown(self):
        self._tmp.cleanup()

    def _run_trials(self):
        run_trials(
            self.trials_dir,
            self.video,
            self.gt,
            infer_script=FAKE_INFER,
            inference_cwd=self.work,
            iou_thresh=0.5,
            eval_plots=False,
            top_params_k=TOP_K,
        )

    def _leaderboard(self) -> list[dict]:
        return json.loads((self.trials_dir / "trial_leaderboard.json").read_text())

    def test_two_phase_real_eval(self):
        # --- Phase 1: exploration ---
        run_exploration(self.base, self.trials_dir, n=N_EXPLORE, seed=SEED)
        explore_files = sorted(self.trials_dir.glob("explore_*.yaml"))
        self.assertEqual(len(explore_files), N_EXPLORE)
        self.assertTrue((self.trials_dir / "manifest.json").is_file())

        self._run_trials()

        # Champion artifacts.
        self.assertTrue((self.trials_dir / "best_config.yaml").is_file())
        manifest = json.loads((self.trials_dir / "manifest.json").read_text())

        top = json.loads((self.trials_dir / "top_params.json").read_text())
        self.assertEqual(len(top), TOP_K)
        for seed_entry in top:
            self.assertEqual(seed_entry["params"], manifest[seed_entry["trial"]])

        lb1 = self._leaderboard()
        self.assertEqual(len(lb1), N_EXPLORE)
        scores1 = [row["score"] for row in lb1]
        self.assertEqual(scores1, sorted(scores1, reverse=True))  # ranked desc
        phase1_best = scores1[0]

        # --- Phase 2: exploitation (warm-start) ---
        run_exploitation(
            self.base,
            self.trials_dir / "top_params.json",
            self.trials_dir,
            n=N_EXPLOIT,
            sigma_frac=0.15,
            top_k=TOP_K,
            seed=SEED,
        )
        exploit_files = sorted(self.trials_dir.glob("exploit_*.yaml"))
        self.assertEqual(len(exploit_files), N_EXPLOIT)

        self._run_trials()

        lb2 = self._leaderboard()
        # Phase-2 leaderboard re-globs explore_ + exploit_, so it is a superset.
        self.assertEqual(len(lb2), N_EXPLORE + N_EXPLOIT)
        phase2_best = lb2[0]["score"]
        self.assertGreaterEqual(phase2_best, phase1_best)

        # Convergence: exploitation should cluster closer to the true optimum.
        manifest = json.loads((self.trials_dir / "manifest.json").read_text())
        self.assertLessEqual(_mean_distance(manifest, "exploit_"), _mean_distance(manifest, "explore_"))

    def test_determinism(self):
        """Same seed -> identical champion seeds across two fresh workspaces."""

        def phase1_top(workdir: Path) -> str:
            trials = workdir / "trials"
            gt = workdir / "ground_truth.csv"
            scene.write_gt_csv(gt, scene.N_FRAMES)
            base = workdir / "base.yaml"
            shutil.copyfile(BASE_YAML, base)
            video = workdir / "clip.mp4"
            video.touch()
            run_exploration(base, trials, n=N_EXPLORE, seed=SEED)
            run_trials(
                trials, video, gt,
                infer_script=FAKE_INFER, inference_cwd=workdir,
                iou_thresh=0.5, eval_plots=False, top_params_k=TOP_K,
            )
            return (trials / "top_params.json").read_text()

        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            self.assertEqual(phase1_top(Path(a)), phase1_top(Path(b)))


if __name__ == "__main__":
    unittest.main()
