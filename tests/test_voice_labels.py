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
        self.assertIn("unavailable", warning)

    def test_matches_technique_spoken_as_separate_words(self):
        words = [
            TranscriptWord("o", 4.0, 4.1),
            TranscriptWord("soto", 4.1, 4.3),
            TranscriptWord("gari", 4.3, 4.5),
        ]

        suggestions = suggest_techniques(words, [("e-1", 5.0, 8.0)])

        self.assertEqual(suggestions["e-1"].predlog_tehnike, "O-soto-gari")
        self.assertEqual(suggestions["e-1"].source_phrase, "o soto gari")


if __name__ == "__main__":
    unittest.main()
