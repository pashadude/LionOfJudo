import unittest

from pipeline.video_review_audio import find_tap_triplet_candidates


class TapTripletCandidateTests(unittest.TestCase):
    def test_finds_three_transients_with_unequal_but_short_gaps(self):
        candidates = find_tap_triplet_candidates([10.00, 10.27, 10.73, 18.0, 19.0])

        self.assertEqual(candidates, [(10.00, 10.27, 10.73)])


if __name__ == "__main__":
    unittest.main()
