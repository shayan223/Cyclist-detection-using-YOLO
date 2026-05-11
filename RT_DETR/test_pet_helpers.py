import importlib.util
import pathlib
import sys
import types
import unittest
from collections import deque

import numpy as np


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

    cv2_stub = types.ModuleType("cv2")
    cv2_stub.FONT_HERSHEY_SIMPLEX = 0
    cv2_stub.LINE_AA = 16
    cv2_stub.arrowedLine = lambda *args, **kwargs: None
    sys.modules.setdefault("cv2", cv2_stub)


_install_test_stubs()


MODULE_PATH = pathlib.Path(__file__).with_name("PET_deepSORT.py")
SPEC = importlib.util.spec_from_file_location("pet_module", MODULE_PATH)
PET = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PET)


class PetHelperTests(unittest.TestCase):
    def test_non_pet_default_seconds_is_high(self):
        self.assertEqual(PET.DEFAULT_NON_PET_SECONDS, 10.0)

    def test_pet_bin_thresholds(self):
        self.assertEqual(PET._pet_bin_label(0.0), "0-1.5s")
        self.assertEqual(PET._pet_bin_label(1.4999), "0-1.5s")
        self.assertEqual(PET._pet_bin_label(1.5), "1.5-3s")
        self.assertEqual(PET._pet_bin_label(3.0), "3-5s")
        self.assertEqual(PET._pet_bin_label(5.0), "5+s")

    def test_pet_bin_risk_thresholds(self):
        self.assertEqual(PET._pet_bin_risk(0.0), 1.0)
        self.assertEqual(PET._pet_bin_risk(1.4999), 1.0)
        self.assertEqual(PET._pet_bin_risk(1.5), 0.75)
        self.assertEqual(PET._pet_bin_risk(3.0), 0.5)
        self.assertEqual(PET._pet_bin_risk(5.0), 0.25)

    def test_pet_bin_plot_level_thresholds(self):
        self.assertEqual(PET._pet_bin_plot_level(0.0), 4)
        self.assertEqual(PET._pet_bin_plot_level(1.4999), 4)
        self.assertEqual(PET._pet_bin_plot_level(1.5), 3)
        self.assertEqual(PET._pet_bin_plot_level(3.0), 2)
        self.assertEqual(PET._pet_bin_plot_level(5.0), 1)

    def test_opposing_filter_keeps_only_opposite_direction(self):
        self.assertFalse(PET._is_opposing_direction(5.0, 15.0))
        self.assertTrue(PET._is_opposing_direction(175.0, 15.0))
        self.assertTrue(PET._is_opposing_direction(180.0, 15.0))
        self.assertFalse(PET._is_opposing_direction(90.0, 15.0))

    def test_collinear_filter_catches_same_and_opposite_direction(self):
        self.assertTrue(PET._is_collinear_direction(5.0, 15.0))
        self.assertTrue(PET._is_collinear_direction(175.0, 15.0))
        self.assertFalse(PET._is_collinear_direction(90.0, 15.0))

    def test_recent_heading_uses_newest_distinct_points(self):
        points = [(0, (0.0, 0.0)), (10, (50.0, 0.0)), (11, (50.0, 5.0))]
        self.assertEqual(PET._recent_heading_vector_from_points(points), (0.0, 5.0))

    def test_buffered_heading_uses_average_over_recent_window(self):
        points = [(0, (0.0, 0.0)), (10, (10.0, 0.0)), (20, (10.0, 10.0))]
        self.assertEqual(PET._buffered_heading_vector_from_points(points, frame_window=15), (0.0, 10.0))
        self.assertEqual(PET._buffered_heading_vector_from_points(points, frame_window=25), (10.0, 10.0))

    def test_buffered_heading_accepts_deque_history(self):
        points = deque([(0, (0.0, 0.0)), (10, (10.0, 0.0)), (20, (10.0, 10.0))])
        self.assertEqual(PET._buffered_heading_vector_from_points(points, frame_window=15), (0.0, 10.0))

    def test_perpendicular_crossing_is_outside_collinear_buffer(self):
        biker = PET._buffered_heading_vector_from_points(
            [(0, (5.0, 10.0)), (5, (5.0, 0.0))],
            frame_window=15,
        )
        pedestrian = PET._buffered_heading_vector_from_points(
            [(0, (0.0, 5.0)), (5, (10.0, 5.0))],
            frame_window=15,
        )
        angle_delta = PET._angle_delta_degrees(biker, pedestrian)
        self.assertAlmostEqual(angle_delta, 90.0, places=4)
        self.assertFalse(PET._is_collinear_direction(angle_delta, 15.0))

    def test_track_anchor_footprint_stays_inside_bbox(self):
        self.assertEqual(PET._track_anchor_point((10.0, 20.0, 30.0, 60.0)), (20.0, 60.0))
        self.assertEqual(
            PET._track_anchor_footprint((10.0, 20.0, 30.0, 60.0)),
            (16.0, 52.0, 24.0, 60.0),
        )

    def test_opposing_motion_relation_labels_toward_and_away(self):
        points_a_toward = [(0, (0.0, 0.0)), (1, (1.0, 0.0))]
        points_b_toward = [(0, (3.0, 0.0)), (1, (2.0, 0.0))]
        self.assertEqual(
            PET._opposing_motion_relation(points_a_toward, points_b_toward, (1.0, 0.0), (-1.0, 0.0)),
            "toward",
        )

        points_a_away = [(0, (1.0, 0.0)), (1, (0.0, 0.0))]
        points_b_away = [(0, (2.0, 0.0)), (1, (3.0, 0.0))]
        self.assertEqual(
            PET._opposing_motion_relation(points_a_away, points_b_away, (-1.0, 0.0), (1.0, 0.0)),
            "away",
        )

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

    def test_polyline_intersection_uses_anchor_footprints(self):
        points_a = [(0, (0.0, 0.0)), (1, (0.0, 10.0))]
        points_b = [(0, (0.4, 0.0)), (1, (0.4, 10.0))]
        footprints_a = [(0, (-0.5, -0.5, 0.5, 0.5)), (1, (-0.5, 9.5, 0.5, 10.5))]
        footprints_b = [(0, (0.2, -0.5, 0.6, 0.5)), (1, (0.2, 9.5, 0.6, 10.5))]
        intersects, point = PET._polyline_intersects_in_rect(
            points_a,
            points_b,
            (-1.0, -1.0, 1.0, 11.0),
            footprints_a,
            footprints_b,
        )
        self.assertTrue(intersects)
        self.assertIsNotNone(point)

    def test_conflict_zone_cells_include_partial_overlap(self):
        polygon = [(5.0, 5.0), (15.0, 5.0), (15.0, 15.0), (5.0, 15.0)]
        cells = PET._conflict_zone_cells_from_polygon(
            polygon, width=20, height=20, grid_rows=2, grid_cols=2
        )
        self.assertEqual(cells, {(0, 0), (0, 1), (1, 0), (1, 1)})

    def test_conflict_zone_cells_exclude_non_overlapping_cells(self):
        polygon = [(1.0, 1.0), (4.0, 1.0), (4.0, 4.0), (1.0, 4.0)]
        cells = PET._conflict_zone_cells_from_polygon(
            polygon, width=20, height=20, grid_rows=2, grid_cols=2
        )
        self.assertEqual(cells, {(0, 0)})

    def test_pet_activation_display_cells_preserve_region_with_conflict_zone(self):
        region_cells = {(0, 0), (0, 1), (1, 0), (1, 1)}
        conflict_zone_cells = {(0, 1), (1, 1)}
        self.assertEqual(
            PET._pet_activation_display_cells((0, 0), region_cells, conflict_zone_cells),
            region_cells,
        )

    def test_pet_activation_display_cells_use_primary_without_conflict_zone(self):
        region_cells = {(0, 0), (0, 1), (1, 0), (1, 1)}
        self.assertEqual(
            PET._pet_activation_display_cells((0, 0), region_cells),
            {(0, 0)},
        )

    def test_speed_conversion(self):
        history = [(0, (0.0, 0.0)), (5, (10.0, 0.0))]
        pixel_speed = PET._estimate_speed_ft_per_sec(history, fps=30)
        self.assertAlmostEqual(pixel_speed, 60.0, places=4)
        self.assertAlmostEqual(PET._speed_to_unit(10.0, "mph"), 6.81818, places=4)
        self.assertAlmostEqual(PET._speed_to_unit(10.0, "ft/s"), 10.0, places=4)

    def test_identity_homography_preserves_speed_math(self):
        H = np.eye(3)
        points = [
            (0, PET._transform_point_homography((0.0, 0.0), H)),
            (5, PET._transform_point_homography((10.0, 0.0), H)),
        ]
        self.assertAlmostEqual(PET._estimate_speed_ft_per_sec(points, fps=30), 60.0, places=4)

    def test_scaled_homography_changes_pixel_speed(self):
        H = np.array([[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 1.0]])
        points = [
            (0, PET._transform_point_homography((0.0, 0.0), H)),
            (5, PET._transform_point_homography((10.0, 0.0), H)),
        ]
        self.assertAlmostEqual(PET._estimate_speed_ft_per_sec(points, fps=30), 120.0, places=4)

    def test_warp_calibration_line_sets_feet_per_pixel_in_warp_space(self):
        calibration = {
            "start": (0.0, 0.0),
            "end": (10.0, 0.0),
            "length_pixels": 10.0,
            "feet_per_pixel": 5.2,
        }
        H = np.array([[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 1.0]])
        context, warning = PET._build_speed_context(calibration, "warp", H)
        self.assertIsNone(warning)
        self.assertEqual(context["space"], "warp")
        self.assertAlmostEqual(context["calibration_length_pixels"], 20.0, places=4)
        self.assertAlmostEqual(context["feet_per_pixel"], 2.6, places=4)

    def test_speed_context_auto_falls_back_to_image_without_homography(self):
        calibration = {
            "start": (0.0, 0.0),
            "end": (10.0, 0.0),
            "length_pixels": 10.0,
            "feet_per_pixel": 5.2,
        }
        context, warning = PET._build_speed_context(calibration, "auto", None)
        self.assertEqual(context["space"], "image")
        self.assertAlmostEqual(context["feet_per_pixel"], 5.2, places=4)
        self.assertIn("falling back", warning)

    def test_speed_context_warp_disables_without_homography(self):
        calibration = {
            "start": (0.0, 0.0),
            "end": (10.0, 0.0),
            "length_pixels": 10.0,
            "feet_per_pixel": 5.2,
        }
        context, warning = PET._build_speed_context(calibration, "warp", None)
        self.assertIsNone(context)
        self.assertIn("homography is unavailable", warning)


if __name__ == "__main__":
    unittest.main()
