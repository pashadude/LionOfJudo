"""Whisper transcript parsing and coach-review technique suggestions."""

import json
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping


MAX_PHRASE_GAP_S = 0.75


@dataclass(frozen=True)
class TranscriptWord:
    text: str
    start_s: float
    end_s: float

    @property
    def midpoint_s(self) -> float:
        return (self.start_s + self.end_s) / 2


@dataclass(frozen=True)
class TechniqueSuggestion:
    predlog_tehnike: str | None
    source_phrase: str | None = None
    confidence: float = 0.0
    user_confirmed: bool = False


TECHNIQUE_VOCABULARY: dict[str, str] = {
    "o soto gari": "O-soto-gari",
    "o-soto-gari": "O-soto-gari",
    "osoto gari": "O-soto-gari",
    "osoto-gari": "O-soto-gari",
    "seoi nage": "Seoi-nage",
    "seoi-nage": "Seoi-nage",
    "ippon seoi nage": "Ippon-seoi-nage",
    "ippon seoi-nage": "Ippon-seoi-nage",
    "o goshi": "O-goshi",
    "o-goshi": "O-goshi",
    "ogoshi": "O-goshi",
    "uki goshi": "Uki-goshi",
    "uki-goshi": "Uki-goshi",
}


def _normalized_phrase(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip().lower().replace("–", "-")).strip(" .,!?:;")


def _word_value(raw: Mapping[str, Any]) -> str:
    return str(raw.get("word", raw.get("text", ""))).strip()


def parse_whisper_json(payload: Mapping[str, Any]) -> list[TranscriptWord]:
    """Parse Whisper JSON words, using segment bounds when word bounds are absent."""
    parsed: list[TranscriptWord] = []
    for segment in payload.get("segments", []):
        segment_start = float(segment.get("start", 0.0))
        segment_end = float(segment.get("end", segment_start))
        raw_words = segment.get("words")
        if raw_words:
            for raw_word in raw_words:
                text = _word_value(raw_word)
                if not text:
                    continue
                start = float(raw_word.get("start", segment_start))
                end = float(raw_word.get("end", segment_end))
                parsed.append(TranscriptWord(text, start, end))
        else:
            text = str(segment.get("text", "")).strip()
            if text:
                parsed.append(TranscriptWord(text, segment_start, segment_end))
    return parsed


def load_whisper_json(transcript_path: Path) -> list[TranscriptWord]:
    with transcript_path.open(encoding="utf-8") as handle:
        return parse_whisper_json(json.load(handle))


def transcribe_with_whisper(video_path: Path) -> tuple[list[TranscriptWord], str | None]:
    """Run the optional Whisper CLI, returning words and a non-fatal warning."""
    video_path = Path(video_path)
    whisper = shutil.which("whisper")
    if whisper is None:
        return [], "Whisper CLI nije dostupan; predlozi tehnika su preskoceni."

    with tempfile.TemporaryDirectory(prefix="lion-whisper-") as output_dir:
        subprocess.run(
            [whisper, str(video_path), "--output_format", "json", "--output_dir", output_dir],
            check=True,
            capture_output=True,
            text=True,
        )
        transcript_path = Path(output_dir) / f"{video_path.stem}.json"
        return load_whisper_json(transcript_path), None


def _matching_technique(text: str) -> str | None:
    return TECHNIQUE_VOCABULARY.get(_normalized_phrase(text))


def _vocabulary_matches(words: list[TranscriptWord]) -> list[tuple[TranscriptWord, str]]:
    matches: list[tuple[TranscriptWord, str]] = []
    words = sorted(words, key=lambda word: word.start_s)
    for start in range(len(words)):
        for width in range(1, min(3, len(words) - start) + 1):
            phrase_words = words[start : start + width]
            if any(
                current.start_s - previous.end_s > MAX_PHRASE_GAP_S
                for previous, current in zip(phrase_words, phrase_words[1:])
            ):
                break
            technique = _matching_technique(" ".join(word.text for word in phrase_words))
            if technique is not None:
                matches.append(
                    (
                        TranscriptWord(
                            " ".join(word.text for word in phrase_words),
                            phrase_words[0].start_s,
                            phrase_words[-1].end_s,
                        ),
                        technique,
                    )
                )
    return matches


def suggest_techniques(
    words: Iterable[TranscriptWord],
    event_windows: Iterable[tuple[str, float, float]],
) -> dict[str, TechniqueSuggestion]:
    """Suggest the closest vocabulary match in each event's allowed time window."""
    vocabulary_words = _vocabulary_matches(list(words))
    suggestions: dict[str, TechniqueSuggestion] = {}
    for event_id, event_start_s, event_end_s in event_windows:
        event_midpoint = (event_start_s + event_end_s) / 2
        nearby = [
            (word, technique)
            for word, technique in vocabulary_words
            if event_start_s - 8 <= word.midpoint_s <= event_end_s + 3
        ]
        if not nearby:
            suggestions[event_id] = TechniqueSuggestion(predlog_tehnike=None)
            continue
        word, technique = min(nearby, key=lambda item: abs(item[0].midpoint_s - event_midpoint))
        distance = abs(word.midpoint_s - event_midpoint)
        confidence = max(0.0, 1.0 - distance / 11.0)
        suggestions[event_id] = TechniqueSuggestion(
            predlog_tehnike=technique,
            source_phrase=word.text,
            confidence=confidence,
        )
    return suggestions
