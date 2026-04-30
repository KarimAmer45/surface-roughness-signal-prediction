from __future__ import annotations

import unittest

import numpy as np

from roughness_prediction.features import extract_features


class FeatureExtractionTest(unittest.TestCase):
    def test_extract_features_returns_expected_signal_metrics(self) -> None:
        sample_rate = 1000
        t = np.linspace(0, 1, sample_rate, endpoint=False)
        y = np.sin(2 * np.pi * 50 * t)

        features = extract_features(y, sample_rate)

        self.assertAlmostEqual(features["duration_s"], 1.0)
        self.assertGreater(features["rms"], 0.6)
        self.assertLess(abs(features["dominant_frequency_hz"] - 50), 2.0)
        self.assertIn("spectral_centroid_hz", features)


if __name__ == "__main__":
    unittest.main()
