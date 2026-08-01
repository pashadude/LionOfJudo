import unittest
from unittest import mock

from pipeline.voice_labels import TranscriptWord, suggest_techniques, transcribe_with_whisper


class VoiceLabelSuggestionTests(unittest.TestCase):
    def test_assigns_nearby_spoken_technique_to_event(self):
        words = [
            TranscriptWord("radimo", 4.0, 4.4),
            TranscriptWord("o-soto-gari", 4.5, 5.1),
        ]

        suggestions = suggest_techniques(words, [("e-1", 5.0, 8.0)])

        self.assertEqual(suggestions["e-1"].predlog_tehnike, "O-soto-gari")
        self.assertEqual(suggestions["e-1"].source_phrase, "o-soto-gari")
        self.assertFalse(suggestions["e-1"].user_confirmed)

    def test_leaves_event_blank_when_no_vocabulary_match_is_nearby(self):
        suggestions = suggest_techniques(
            [TranscriptWord("pozdrav", 0.0, 0.5)], [("e-1", 8.0, 10.0)]
        )

        self.assertIsNone(suggestions["e-1"].predlog_tehnike)

    def test_normalizes_spelling_variants_and_uses_segment_fallback(self):
        from pipeline.voice_labels import parse_whisper_json

        words = parse_whisper_json(
            {
                "segments": [
                    {"start": 2.0, "end": 3.0, "text": "seoi nage"},
                    {"start": 4.0, "end": 5.0, "words": [{"word": "osoto gari"}]},
                ]
            }
        )

        self.assertEqual(words[0], TranscriptWord("seoi nage", 2.0, 3.0))
        self.assertEqual(words[1], TranscriptWord("osoto gari", 4.0, 5.0))
        self.assertEqual(
            suggest_techniques(words, [("e-1", 2.0, 2.5)])[
                "e-1"
            ].predlog_tehnike,
            "Seoi-nage",
        )

    @mock.patch("pipeline.voice_labels.shutil.which", return_value=None)
    def test_warns_and_returns_empty_transcript_without_whisper(self, _which):
        words, warning = transcribe_with_whisper("training.mp4")

        self.assertEqual(words, [])
        self.assertEqual(
            warning,
            "Whisper CLI nije dostupan; predlozi tehnika su preskoceni.",
        )

    def test_suggestion_maps_source_evidence_to_canonical_review_fields(self):
        suggestion = suggest_techniques(
            [TranscriptWord("o soto gari", 11.5, 12.2)],
            [("e-1", 11.0, 14.0)],
        )["e-1"]

        review_fields = suggestion.to_review_fields()

        self.assertEqual(review_fields["predlog_tehnike"], "O-soto-gari")
        self.assertEqual(review_fields["glasovna_fraza"], "o soto gari")
        self.assertGreater(review_fields["pouzdanost_glasa"], 0.0)
        self.assertEqual(review_fields["glasovna_fraza_pocetak_s"], 11.5)
        self.assertEqual(review_fields["glasovna_fraza_kraj_s"], 12.2)
        self.assertNotIn("source_phrase", review_fields)
        self.assertNotIn("confidence", review_fields)

    def test_matches_technique_spoken_as_separate_words(self):
        words = [
            TranscriptWord("o", 4.0, 4.1),
            TranscriptWord("soto", 4.1, 4.3),
            TranscriptWord("gari", 4.3, 4.5),
        ]

        suggestions = suggest_techniques(words, [("e-1", 5.0, 8.0)])

        self.assertEqual(suggestions["e-1"].predlog_tehnike, "O-soto-gari")
        self.assertEqual(suggestions["e-1"].source_phrase, "o soto gari")

    def test_does_not_join_technique_words_across_a_long_pause(self):
        words = [
            TranscriptWord("o", 4.0, 4.1),
            TranscriptWord("soto", 4.1, 4.2),
            TranscriptWord("gari", 6.0, 6.1),
        ]

        suggestions = suggest_techniques(words, [("e-1", 5.0, 8.0)])

        self.assertIsNone(suggestions["e-1"].predlog_tehnike)


if __name__ == "__main__":
    unittest.main()
