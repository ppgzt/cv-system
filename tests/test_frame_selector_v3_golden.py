"""Golden pequeno do artefato e preprocessing operacional Frame Selector v3."""

from __future__ import annotations

import hashlib
import unittest
from pathlib import Path

import numpy as np
import skimage.io as ski
import tensorflow as tf

from domain.modules.frame_selection import FrameSelection


ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "infra/models/frame_selector.tflite"
EXPECTED_MODEL_SHA256 = "f0886d0f01a1b48ccb836da7ea139caa58f0e0e445ee27ef2ec2a07abd9adca7"

GOLDEN_FRAMES = (
    ("suited", "01mf", "1mf_14196_2025_01_25_17_42_41_145102_DEPTH_320_240_1.png", 0.9999750852584839, True),
    ("parcial", "0278", "0278_15336_2025_01_26_09_06_09_108098_DEPTH_320_240_1.png", 3.864908285322599e-05, False),
    ("background", "0014s2", "0014_2406_2025_01_25_10_48_38_481264_DEPTH_320_240_1.png", 3.067339321205509e-06, False),
    ("ruido", "0909", "909_16125_2025_01_26_09_10_51_045824_DEPTH_320_240_1.png", 1.1101371910626767e-07, False),
)


class FrameSelectorV3GoldenTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.selector = FrameSelection(str(MODEL), threshold=0.5, num_threads=1)

    def test_active_artifact_and_runtime_tensor_contract(self):
        digest = hashlib.sha256(MODEL.read_bytes()).hexdigest()
        self.assertEqual(digest, EXPECTED_MODEL_SHA256)
        self.assertEqual(self.selector._input_tensor.shape, (1, 224, 224, 3))
        self.assertEqual(self.selector._input_tensor.dtype, np.float32)
        self.assertEqual(self.selector.threshold, 0.5)

    def test_roi10_clip_before_bilinear_resize_and_normalization(self):
        raw = np.arange(240 * 320, dtype=np.uint32).reshape(240, 320)
        raw = np.minimum(raw, 3000).astype(np.uint16)
        actual = self.selector._preprocess_fn(raw[:, :, np.newaxis]).numpy()

        crop = raw[24:216, 32:288].astype(np.float32)
        crop = np.clip(crop, 0.0, 1950.0)[:, :, np.newaxis]
        expected = tf.image.resize(crop, [224, 224], method="bilinear")
        expected = tf.image.grayscale_to_rgb(expected)
        expected = 2.0 * (expected / 1950.0) - 1.0

        self.assertEqual(actual.shape, (224, 224, 3))
        self.assertEqual(actual.dtype, np.float32)
        self.assertTrue(np.isfinite(actual).all())
        self.assertGreaterEqual(float(actual.min()), -1.0)
        self.assertLessEqual(float(actual.max()), 1.0)
        np.testing.assert_allclose(actual, expected.numpy(), rtol=0.0, atol=1e-6)

    def test_real_suited_partial_background_noise_probabilities(self):
        for label, passage, filename, expected_probability, expected_decision in GOLDEN_FRAMES:
            with self.subTest(label=label, passage=passage):
                path = ROOT / "data/exp1/DEPTH" / passage / filename
                raw = ski.imread(path)
                self.assertEqual(raw.shape, (240, 320))
                self.assertEqual(raw.dtype, np.uint16)

                processed = self.selector._preprocess_fn(
                    self.selector._to_single_channel(raw)
                ).numpy()
                self.selector._input_tensor[0] = processed
                self.assertEqual(self.selector._input_tensor.shape, (1, 224, 224, 3))
                self.assertEqual(self.selector._input_tensor.dtype, np.float32)
                self.assertTrue(np.isfinite(self.selector._input_tensor).all())
                self.assertGreaterEqual(float(self.selector._input_tensor.min()), -1.0)
                self.assertLessEqual(float(self.selector._input_tensor.max()), 1.0)

                probability = self.selector.predict(raw)
                self.assertAlmostEqual(probability, expected_probability, delta=5e-5)
                self.assertEqual(probability > 0.5, expected_decision)


if __name__ == "__main__":
    unittest.main()
