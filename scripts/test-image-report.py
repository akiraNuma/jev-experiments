"""独立した既知の矩形で、位置評価と対になった比較を検証する。"""
import importlib.util
import unittest
from pathlib import Path

spec = importlib.util.spec_from_file_location("report", Path(__file__).with_name("recipe-images-report.py"))
report = importlib.util.module_from_spec(spec)
spec.loader.exec_module(report)


class ImageMetricsTest(unittest.TestCase):
    def test_overlap_and_disjoint(self):
        square = {"left": 0, "top": 0, "right": 0.5, "bottom": 0.5}
        other = {"left": 0.25, "top": 0, "right": 0.75, "bottom": 0.5}
        disjoint = {"left": 0.5, "top": 0.5, "right": 1, "bottom": 1}
        self.assertEqual(report.iou(square, square), 1)
        self.assertAlmostEqual(report.iou(square, other), 1 / 3)
        self.assertEqual(report.iou(square, disjoint), 0)

    def test_containment(self):
        full = {"left": 0, "top": 0, "right": 1, "bottom": 1}
        self.assertAlmostEqual(report.iou(full, report.CENTER), 0.25)

    def test_paired_difference_and_no_difference(self):
        a = {i: True for i in range(6)}
        b = {i: False for i in range(6)}
        self.assertEqual(report.mcnemar(a, b)["p_exact_two_sided"], 0.03125)
        self.assertEqual(report.mcnemar(a, a)["p_exact_two_sided"], 1)

    def test_small_sample_uncertainty(self):
        low, high = report.wilson(60, 60)
        self.assertAlmostEqual(high, 1)
        self.assertLess(low, 0.95)
        self.assertGreater(low, 0.93)


if __name__ == "__main__":
    unittest.main()
