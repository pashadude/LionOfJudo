import unittest

from pipeline.video_review_audio import candidate_details, find_tap_triplet_candidates


class TapTripletCandidateTests(unittest.TestCase):
    def test_finds_three_transients_with_unequal_but_short_gaps(self):
        candidates = find_tap_triplet_candidates([10.00, 10.27, 10.73, 18.0, 19.0])

        self.assertEqual(candidates, [(10.00, 10.27, 10.73)])

    def test_retains_per_candidate_confidence_from_peak_prominences(self):
        candidates = candidate_details([10.00, 10.27, 10.73], [2.0, 4.0, 8.0])

        self.assertEqual(candidates[0].peaks_s, (10.0, 10.27, 10.73))
        self.assertAlmostEqual(candidates[0].confidence, 14.0 / 24.0)
        self.assertFalse(candidates[0].user_confirmed)


if __name__ == "__main__":
    unittest.main()
