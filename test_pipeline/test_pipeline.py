import unittest
import numpy as np
from pipeline_math_profile import (
    TASK_COMPLETION_RATE_AVG,
    simulate_exam_session,
    build_primary_to_secondary_scale,
    validate_data_and_scale,
    calculate_psychometrics_tz,
    detect_anomalies_tukey,
)


class TestAnalyticsPipeline(unittest.TestCase):
    def test_scale_monotonicity_and_bounds(self):
        scale_df = build_primary_to_secondary_scale(TASK_COMPLETION_RATE_AVG)
        sec = scale_df["Вторичный балл"].to_numpy()
        self.assertEqual(len(sec), 13)
        self.assertEqual(sec[0], 0)
        self.assertEqual(sec[-1], 70)
        self.assertTrue(np.all(np.diff(sec) >= 0), "Шкала должна быть неубывающей на всём диапазоне 0..12")

    def test_reproducibility(self):
        s1, _ = simulate_exam_session(TASK_COMPLETION_RATE_AVG, n_students=5000, seed=123)
        s2, _ = simulate_exam_session(TASK_COMPLETION_RATE_AVG, n_students=5000, seed=123)
        self.assertTrue(np.array_equal(s1, s2), "Повторный прогон обязан давать идентичный результат (NFR-01)")

    def test_psychometrics_and_kelley_27(self):
        scores, probs = simulate_exam_session(TASK_COMPLETION_RATE_AVG, n_students=10000, seed=42)
        scale_df = build_primary_to_secondary_scale(TASK_COMPLETION_RATE_AVG)
        self.assertTrue(validate_data_and_scale(scores, scale_df))

        alpha, report_df, flagged_df, tasks_json = calculate_psychometrics_tz(scores)
        self.assertTrue(0.45 <= alpha <= 0.85)
        self.assertEqual(len(report_df), 12)
        self.assertIn("D (индекс дифф.)", report_df.columns)
        self.assertTrue(all(0.0 <= t["pValue"] <= 1.0 for t in tasks_json))

    def test_tukey_anomalies_no_pii(self):
        scores, probs = simulate_exam_session(TASK_COMPLETION_RATE_AVG, n_students=2000, seed=42)
        scores[0] = np.array([0, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1])
        scale_df = build_primary_to_secondary_scale(TASK_COMPLETION_RATE_AVG)
        res = detect_anomalies_tukey(scores, probs, scale_df)
        self.assertGreaterEqual(res["anomaliesDetectedTotal"], 1)
        for card in res["cards"]:
            self.assertNotIn("fio", card)
            self.assertNotIn("passport", card)


if __name__ == "__main__":
    unittest.main()
