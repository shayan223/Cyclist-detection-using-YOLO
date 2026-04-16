import importlib.util
import pathlib
import sys
import types
import unittest


def _install_test_stubs():
    pandas_stub = types.ModuleType("pandas")
    pandas_stub.DataFrame = object
    sys.modules.setdefault("pandas", pandas_stub)

    matplotlib_stub = types.ModuleType("matplotlib")
    matplotlib_stub.use = lambda *args, **kwargs: None
    pyplot_stub = types.ModuleType("matplotlib.pyplot")
    pyplot_stub.subplots = lambda *args, **kwargs: (None, None)
    pyplot_stub.close = lambda *args, **kwargs: None
    sys.modules.setdefault("matplotlib", matplotlib_stub)
    sys.modules.setdefault("matplotlib.pyplot", pyplot_stub)

    ultralytics_stub = types.ModuleType("ultralytics")
    class _DummyRTDETR:
        def __init__(self, *args, **kwargs):
            pass
    ultralytics_stub.RTDETR = _DummyRTDETR
    sys.modules.setdefault("ultralytics", ultralytics_stub)


_install_test_stubs()


MODULE_PATH = pathlib.Path(__file__).with_name("PET_deepSORT.py")
SPEC = importlib.util.spec_from_file_location("pet_module", MODULE_PATH)
PET = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PET)


class PetHelperTests(unittest.TestCase):
    def test_pet_bin_thresholds(self):
        self.assertEqual(PET._pet_bin_label(0.0), "0-1.5s")
        self.assertEqual(PET._pet_bin_label(1.4999), "0-1.5s")
        self.assertEqual(PET._pet_bin_label(1.5), "1.5-3s")
        self.assertEqual(PET._pet_bin_label(3.0), "3-5s")
        self.assertEqual(PET._pet_bin_label(5.0), "5+s")

    def test_parallel_filter_catches_same_and_opposite_direction(self):
        self.assertTrue(PET._is_near_parallel(5.0, 15.0))
        self.assertTrue(PET._is_near_parallel(175.0, 15.0))
        self.assertFalse(PET._is_near_parallel(90.0, 15.0))

    def test_polyline_intersection_inside_region(self):
        points_a = [(0, (0.0, 0.0)), (1, (10.0, 10.0))]
        points_b = [(0, (0.0, 10.0)), (1, (10.0, 0.0))]
        intersects, point = PET._polyline_intersects_in_rect(points_a, points_b, (0.0, 0.0, 10.0, 10.0))
        self.assertTrue(intersects)
        self.assertAlmostEqual(point[0], 5.0, places=3)
        self.assertAlmostEqual(point[1], 5.0, places=3)

    def test_polyline_intersection_outside_region(self):
        points_a = [(0, (0.0, 0.0)), (1, (10.0, 10.0))]
        points_b = [(0, (0.0, 10.0)), (1, (10.0, 0.0))]
        intersects, point = PET._polyline_intersects_in_rect(points_a, points_b, (6.0, 6.0, 10.0, 10.0))
        self.assertFalse(intersects)
        self.assertIsNone(point)

    def test_speed_conversion(self):
        history = [(0, (0.0, 0.0)), (5, (10.0, 0.0))]
        pixel_speed = PET._estimate_speed_ft_per_sec(history, fps=30)
        self.assertAlmostEqual(pixel_speed, 60.0, places=4)
        self.assertAlmostEqual(PET._speed_to_unit(10.0, "mph"), 6.81818, places=4)
        self.assertAlmostEqual(PET._speed_to_unit(10.0, "ft/s"), 10.0, places=4)


if __name__ == "__main__":
    unittest.main()
