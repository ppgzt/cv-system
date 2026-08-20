import importlib.util
import unittest
from collections import deque
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = (
    REPO_ROOT
    / "data-analysis"
    / "visual_event_preprocessing_ablation"
    / "run_ablation.py"
)
SPEC = importlib.util.spec_from_file_location("visual_event_preprocessing_ablation", MODULE_PATH)
ablation = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(ablation)


class PreprocessingTests(unittest.TestCase):
    def test_blurs_preserve_shape_and_produce_float32(self):
        frame = np.arange(35, dtype=np.uint16).reshape(5, 7)
        for kind in ("gaussian_3x3", "gaussian_5x5", "median_3x3", "median_5x5"):
            filtered = ablation.preprocess(frame, kind)
            self.assertEqual(frame.shape, filtered.shape)
            self.assertEqual(np.float32, filtered.dtype)

    def test_absdiff_converts_before_subtraction_and_avoids_uint16_underflow(self):
        previous = np.array([[2600]], dtype=np.uint16)
        current = np.array([[10]], dtype=np.uint16)
        difference = ablation.safe_absdiff(previous, current)
        self.assertEqual(np.float32, difference.dtype)
        self.assertEqual(2590.0, float(difference[0, 0]))

    def test_morphology_operates_on_binary_mask(self):
        isolated = np.zeros((5, 5), dtype=bool)
        isolated[2, 2] = True
        self.assertFalse(np.any(ablation.morph_mask(isolated, "opening_3x3")))

        hole = np.ones((5, 5), dtype=bool)
        hole[2, 2] = False
        self.assertTrue(ablation.morph_mask(hole, "closing_3x3")[2, 2])

    def test_score_smoothing_is_strictly_causal(self):
        history = deque(maxlen=3)
        observed = [ablation.smooth_score(history, value, "mean3") for value in (1.0, 2.0, 100.0)]
        self.assertEqual([1.0, 1.5, 103.0 / 3.0], observed)
        # Nenhum score futuro participou das primeiras duas saídas.
        self.assertEqual(1.0, observed[0])
        self.assertEqual(1.5, observed[1])

    def test_invalid_reset_makes_next_valid_frame_a_baseline(self):
        indexes = {
            "passage": [
                {"label": "background"},
                {"label": "background"},
                {"label": "ruido"},
                {"label": "background"},
                {"label": "background"},
            ]
        }
        p99 = {
            ("passage", 1): 100.0,
            ("passage", 2): 100.0,
            ("passage", 3): 3000.0,
            ("passage", 4): 100.0,
            ("passage", 5): 100.0,
        }
        pairs = {"passage": {2: {"V0_baseline": 0.1}, 4: {"V0_baseline": 0.9}, 5: {"V0_baseline": 0.2}}}
        series = ablation.build_series(indexes, p99, pairs, ["V0_baseline"])
        values = series["predicted_p99"]["V0_baseline"]["passage"]
        self.assertTrue(np.isnan(values[0]))
        self.assertEqual(0.1, values[1])
        self.assertTrue(np.isnan(values[2]))
        self.assertTrue(np.isnan(values[3]))
        self.assertEqual(0.2, values[4])

    def test_temporal_series_never_carries_a_previous_frame_between_passages(self):
        indexes = {
            "first": [{"label": "background"}, {"label": "background"}],
            "second": [{"label": "background"}, {"label": "background"}],
        }
        p99 = {(passage, index): 100.0 for passage in indexes for index in (1, 2)}
        pairs = {
            "first": {2: {"V0_baseline": 0.7}},
            "second": {2: {"V0_baseline": 0.2}},
        }
        series = ablation.build_series(indexes, p99, pairs, ["V0_baseline"])
        values = series["predicted_p99"]["V0_baseline"]
        self.assertTrue(np.isnan(values["first"][0]))
        self.assertTrue(np.isnan(values["second"][0]))
        self.assertEqual(0.7, values["first"][1])
        self.assertEqual(0.2, values["second"][1])

    def test_pdi_score_is_deterministic(self):
        previous = np.zeros((20, 30), dtype=np.uint16)
        current = previous.copy()
        current[5:12, 9:20] = 300
        specification = {"preprocessing": "none", "morphology": "none"}
        first = ablation.pdi_score(previous, current, specification)
        second = ablation.pdi_score(previous, current, specification)
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
