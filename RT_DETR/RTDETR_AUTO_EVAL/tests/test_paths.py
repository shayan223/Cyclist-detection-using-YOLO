import unittest

from rtdetr_eval.paths import default_ground_truth, default_trials_dir


class TestPaths(unittest.TestCase):
    def test_default_trials_dir_shape(self):
        p = default_trials_dir()
        # Ensure we don't accidentally regress to old cam_01 naming.
        self.assertIn("camera_1", str(p))

    def test_default_ground_truth_shape(self):
        p = default_ground_truth()
        self.assertTrue(str(p).endswith(".csv"))

