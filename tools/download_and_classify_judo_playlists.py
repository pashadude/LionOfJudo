#!/usr/bin/env python3
"""Download matched judo playlist videos and hard-link them into dataset folders."""

from __future__ import annotations

import argparse
import csv
import os
import re
import shutil
import sys
import unicodedata
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Iterable

import yt_dlp


MEDIA_EXTENSIONS = {
    ".avi",
    ".m4v",
    ".mkv",
    ".mov",
    ".mp4",
    ".webm",
}

FUZZY_WINDOW_MIN_RATIO = 0.96

PLAYLIST_URLS = [
    "https://www.youtube.com/playlist?list=PLTNkUnUBIfH8KMF4es0VlnSyBajeZ6Dgc",
    "https://www.youtube.com/playlist?list=PLTNkUnUBIfH8eswNVtMFxndzzCtBleKLG",
    "https://www.youtube.com/playlist?list=PLTNkUnUBIfH-extcGyOpUWZbtmywdW29h",
    "https://www.youtube.com/playlist?list=PLTNkUnUBIfH_bmA8i4xq-nSoU4PE9H-d3",
    "https://www.youtube.com/playlist?list=PLTNkUnUBIfH8vLrllj6BDzhQu1arTNXsb",
    "https://www.youtube.com/playlist?list=PLu4FypKU5gsbZNCU254W8FJXINp1R_GyK",
    "https://www.youtube.com/playlist?list=PLu4FypKU5gsbVMojk14TxMBrUMgSpDXs8",
    "https://www.youtube.com/playlist?list=PLu4FypKU5gsaCn69XR6xdKdM58rcnMwyU",
    "https://www.youtube.com/playlist?list=PLu4FypKU5gsaQvTIMMopgH7YpdAeipTmv",
    "https://www.youtube.com/playlist?list=PLu4FypKU5gsYspRKAXJCRDO_BIWBFPSvI",
    "https://www.youtube.com/playlist?list=PLnpCVzrtJhqatkEHGI_L1os-gmYnwBrBA",
    "https://www.youtube.com/playlist?list=PLnpCVzrtJhqZdMN3jwZBuHpmF8083SzOE",
    "https://www.youtube.com/playlist?list=PLnpCVzrtJhqaCEb0SJcJkoPxBsInQKKaj",
    "https://www.youtube.com/playlist?list=PLnpCVzrtJhqYU_mA6ilT4lCAvQUifrVu2",
    "https://www.youtube.com/playlist?list=PLnpCVzrtJhqYWiEJ8epvNjPQ7nQyt5F1C",
    "https://www.youtube.com/playlist?list=PL40537E1A227EDD39",
    "https://www.youtube.com/playlist?list=PL7DEB6A04548ABC98",
    "https://www.youtube.com/playlist?list=PL_yLXK1vk_nPvx32vn1035lcJYX23Qfsw",
    "https://www.youtube.com/playlist?list=PL_yLXK1vk_nOiOjAgthPVX1criHEAUVzQ",
    "https://www.youtube.com/playlist?list=PL_yLXK1vk_nOQYxHdrrUsz1MT9UK_BRmK",
    "https://www.youtube.com/playlist?list=PL_yLXK1vk_nPY7jO7TERv0-Bva5-zf5Je",
    "https://www.youtube.com/playlist?list=PL_yLXK1vk_nOF57lfHVaiLvZvqUKsF7DE",
    "https://www.youtube.com/playlist?list=PL_yLXK1vk_nNNe0XPALuZIt9d3fLSwdXV",
    "https://www.youtube.com/playlist?list=PL_yLXK1vk_nOTgZDvINEGjZnnqaK4Kfvq",
    "https://www.youtube.com/playlist?list=PL_yLXK1vk_nPVr8mYA2W7tRuz0uMXHLNb",
]


@dataclass(frozen=True)
class Technique:
    folder: str
    phrase: str
    tokens: tuple[str, ...]


@dataclass(frozen=True)
class Match:
    folder: str
    score: float
    reason: str
    tokens: tuple[str, ...]


@dataclass(frozen=True)
class PlannedVideo:
    playlist_id: str
    playlist_title: str
    video_id: str
    title: str
    url: str
    matches: tuple[Match, ...]


REPLACEMENTS = [
    ("osotogari", "o soto gari"),
    ("osoto gari", "o soto gari"),
    ("osoto", "o soto"),
    ("ouchigari", "o uchi gari"),
    ("ouchi gari", "o uchi gari"),
    ("ko ouchi", "ko uchi"),
    ("ouchi", "o uchi"),
    ("ogoshi", "o goshi"),
    ("kosotogari", "ko soto gari"),
    ("kosoto gari", "ko soto gari"),
    ("kosoto", "ko soto"),
    ("kouchigari", "ko uchi gari"),
    ("kouchi gari", "ko uchi gari"),
    ("kouchi", "ko uchi"),
    ("kuchikidaoshi", "kuchiki taoshi"),
    ("kuchiki daoshi", "kuchiki taoshi"),
    ("kuchikitaoshi", "kuchiki taoshi"),
    ("seoinage", "seoi nage"),
    ("ipponseoi", "ippon seoi"),
    ("moroteseoi", "morote seoi"),
    ("seoi nagi", "seoi nage"),
    ("taiotoshi", "tai otoshi"),
    ("tae otoshi", "tai otoshi"),
    ("tani otosh", "tani otoshi"),
    ("taniotoshi", "tani otoshi"),
    ("uchimata", "uchi mata"),
    ("kataguruma", "kata guruma"),
    ("haraigoshi", "harai goshi"),
    ("hanegoshi", "hane goshi"),
    ("ukigoshi", "uki goshi"),
    ("oguruma", "o guruma"),
    ("tomoe nage", "tomoe nage"),
    ("tomoenage", "tomoe nage"),
    ("yokotomoenage", "yoko tomoe nage"),
    ("sumigaeshi", "sumi gaeshi"),
    ("suminage", "sumi gaeshi"),
    ("sukuinage", "sukui nage"),
    ("morotegari", "morote gari"),
    ("kibisugaeshi", "kibisu gaeshi"),
    ("uranage", "ura nage"),
    ("sodetsurikomigoshi", "sode tsurikomi goshi"),
    ("sasaetsurikomiashi", "sasae tsurikomi ashi"),
    ("tsuri komi", "tsurikomi"),
    ("tsurikomigoshi", "tsurikomi goshi"),
    ("okuriashiharai", "okuri ashi harai"),
    ("okuriashibarai", "okuri ashi barai"),
    ("ashiwaza", "ashi waza"),
    ("ashiharai", "ashi harai"),
    ("ashibarai", "ashi barai"),
    ("ashi harai", "ashi barai"),
    ("deashi harai", "de ashi barai"),
    ("deashibarai", "de ashi barai"),
    ("deashi", "de ashi"),
    ("de ashi harai", "de ashi barai"),
    ("hiza guruma", "hiza guruma"),
    ("uchimatas", "uchi mata"),
    ("makkikomi", "makikomi"),
    ("kata te jime", "katate jime"),
    ("udegeashi", "ude gaeshi"),
    ("udegaeshi", "ude gaeshi"),
]


NOISE_TOKENS = {
    "and",
    "at",
    "basics",
    "bjj",
    "breakdown",
    "by",
    "drill",
    "drills",
    "episode",
    "ep",
    "for",
    "from",
    "grappling",
    "higashi",
    "how",
    "in",
    "instructional",
    "johan",
    "judo",
    "ken",
    "kokushi",
    "lenny",
    "mastery",
    "nyc",
    "of",
    "oshima",
    "seminar",
    "series",
    "shintaro",
    "subiza",
    "the",
    "to",
    "tricks",
    "tutorial",
    "variation",
    "variations",
    "with",
}

TYPO_FOLDER_ALIASES = {
    "tae-otoshi": "tai-otoshi",
    "tamoe-nage": "tomoe-nage",
}

CANONICAL_FOLDER_ALIASES = {
    **TYPO_FOLDER_ALIASES,
    "ankle-block-te-guruma": "te-guruma",
    "arm-weave-sukui-nage": "sukui-nage",
    "cross-body-o-soto-gari": "o-soto-gari",
    "cross-grip-kata-guruma": "kata-guruma",
    "cross-grip-o-soto-gake": "o-soto-gake",
    "crossed-arm-sode-tsurikomi-goshi": "sode-tsurikomi-goshi",
    "daki-ko-soto-gake": "ko-soto-gake",
    "direct-attack-ura-nage": "ura-nage",
    "drop-sode-tsurikomi-goshi": "sode-tsurikomi-goshi",
    "duck-under-tai-otoshi": "tai-otoshi",
    "ducking-kouchi-makikomi": "kouchi-makikomi",
    "fallon-sode-tsurikomi-goshi": "sode-tsurikomi-goshi",
    "furiko-yoko-tomoe-nage": "yoko-tomoe-nage",
    "georgian-grip-kouchi-gari": "kouchi-gari",
    "georgian-grip-o-soto-gake": "o-soto-gake",
    "georgian-grip-uchi-mata-makikomi": "uchi-mata-makikomi",
    "grapevine-sumi-gaeshi": "sumi-gaeshi",
    "huizinga-kata-guruma": "kata-guruma",
    "iliadis-eri-seoi-otoshi": "seoi-otoshi",
    "inside-te-guruma": "te-guruma",
    "ippon-o-soto-gari": "o-soto-gari",
    "kata-hiza-tai-otoshi": "tai-otoshi",
    "ken-ken-o-uchi-gari": "o-uchi-gari",
    "ken-ken-uchi-mata": "uchi-mata",
    "knee-block-seoi-otoshi": "seoi-otoshi",
    "koga-seoi-nage": "seoi-nage",
    "ko-uchi-gari": "kouchi-gari",
    "ko-uchi-makikomi": "kouchi-makikomi",
    "lapel-sasae-tsurikomi-ashi": "sasae-tsurikomi-ashi",
    "mae-ura-nage": "ura-nage",
    "mollaei-kata-guruma": "kata-guruma",
    "nidan-ko-soto-gari": "ko-soto-gari",
    "one-step-uchi-mata": "uchi-mata",
    "outside-kuchiki-taoshi": "kuchiki-taoshi",
    "outside-leg-kata-guruma": "kata-guruma",
    "outside-sumi-gaeshi": "sumi-gaeshi",
    "overhook-uchi-mata": "uchi-mata",
    "overhook-yoko-wakare": "yoko-wakare",
    "reverse-o-goshi": "o-goshi",
    "reverse-seoi-nage": "seoi-nage",
    "rolling-sode-tsurikomi-goshi": "sode-tsurikomi-goshi",
    "ryo-hiza-kata-guruma": "kata-guruma",
    "ryo-hiza-seoi-otoshi": "seoi-otoshi",
    "ryo-sode-tsurikomi-goshi": "sode-tsurikomi-goshi",
    "sacrifice-hiza-guruma": "hiza-guruma",
    "same-side-sumi-gaeshi": "sumi-gaeshi",
    "single-side-kata-guruma": "kata-guruma",
    "sleeve-grip-seoi-nage": "seoi-nage",
    "sleeve-grip-tai-otoshi": "tai-otoshi",
    "soto-ashi-dori-o-uchi-gari": "o-uchi-gari",
    "spinning-hikikomi-gaeshi": "hikikomi-gaeshi",
    "spinning-uchi-mata": "uchi-mata",
    "split-leg-seoi-nage": "seoi-nage",
    "sticky-foot-ko-soto-gake": "ko-soto-gake",
    "sticky-foot-ko-soto-gari": "ko-soto-gari",
    "switch-side-kibisu-gaeshi": "kibisu-gaeshi",
    "switch-side-tani-otoshi": "tani-otoshi",
    "wide-stance-tsurikomi-goshi": "tsurikomi-goshi",
    "yoko-kata-guruma-otoshi": "kata-guruma",
    "zantaraia-uchi-mata": "uchi-mata",
}

KNOWN_THROW_FOLDERS = {
    "ashi-guruma",
    "de-ashi-barai",
    "hane-goshi",
    "hane-makikomi",
    "harai-goshi",
    "harai-makikomi",
    "harai-tsurikomi-ashi",
    "hiza-guruma",
    "hikikomi-gaeshi",
    "ippon-seoi-nage",
    "kata-guruma",
    "kibisu-gaeshi",
    "ko-soto-gake",
    "ko-soto-gari",
    "kouchi-gari",
    "kouchi-makikomi",
    "kuchiki-taoshi",
    "morote-gari",
    "morote-seoi-nage",
    "o-goshi",
    "o-guruma",
    "o-soto-gaeshi",
    "o-soto-gari",
    "o-soto-guruma",
    "o-soto-otoshi",
    "o-uchi-gaeshi",
    "o-uchi-gari",
    "obi-otoshi",
    "obi-tori-gaeshi",
    "okuri-ashi-barai",
    "sasae-tsurikomi-ashi",
    "seoi-nage",
    "seoi-otoshi",
    "sode-tsurikomi-goshi",
    "soto-makikomi",
    "sukui-nage",
    "sumi-gaeshi",
    "sumi-otoshi",
    "tai-otoshi",
    "tani-otoshi",
    "tawara-gaeshi",
    "tomoe-nage",
    "tsubame-gaeshi",
    "tsuri-goshi",
    "tsurikomi-goshi",
    "uchi-makikomi",
    "uchi-mata",
    "uchi-mata-makikomi",
    "uki-goshi",
    "uki-otoshi",
    "uki-waza",
    "ura-nage",
    "ushiro-goshi",
    "utsuri-goshi",
    "yoko-gake",
    "yoko-guruma",
    "yoko-otoshi",
    "yoko-tomoe-nage",
    "yoko-wakare",
}

TECHNIQUE_ALIASES = {
    "ko-soto-gari": (("ko", "soto"),),
    "kouchi-gari": (("ko", "uchi"),),
    "o-soto-gari": (("o", "soto"),),
    "o-uchi-gari": (("o", "uchi"),),
    "sode-tsurikomi-goshi": (("sode",),),
    "tomoe-nage": (("tomoe",),),
    "yoko-tomoe-nage": (("yoko", "tomoe"),),
}

ALIAS_BLOCKED_TOKENS = {
    ("sode-tsurikomi-goshi", ("sode",)): {"jime"},
}


def canonical_folder(folder: str) -> str:
    seen = set()
    while folder in CANONICAL_FOLDER_ALIASES and folder not in seen:
        seen.add(folder)
        folder = CANONICAL_FOLDER_ALIASES[folder]
    return folder


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.lower()
    text = text.replace("&", " and ")
    text = re.sub(r"\[[^\]]+\]$", " ", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    text = f" {text} "
    for old, new in REPLACEMENTS:
        text = text.replace(f" {old} ", f" {new} ")
    return re.sub(r"\s+", " ", text).strip()


def tokens_for(text: str) -> tuple[str, ...]:
    return tuple(normalize_text(text).split())


def useful_title_tokens(title: str) -> tuple[str, ...]:
    return tuple(tok for tok in tokens_for(title) if tok not in NOISE_TOKENS)


def is_subsequence(needle: tuple[str, ...], haystack: tuple[str, ...]) -> bool:
    pos = 0
    for token in haystack:
        if pos < len(needle) and token == needle[pos]:
            pos += 1
    return pos == len(needle)


def contains_phrase(tokens: tuple[str, ...], phrase: tuple[str, ...]) -> bool:
    if not phrase or len(phrase) > len(tokens):
        return False
    size = len(phrase)
    return any(tokens[i : i + size] == phrase for i in range(len(tokens) - size + 1))


def best_window_ratio(title_tokens: tuple[str, ...], phrase: tuple[str, ...]) -> float:
    if not title_tokens or not phrase:
        return 0.0
    low = max(1, len(phrase) - 1)
    high = min(len(title_tokens), len(phrase) + 1)
    phrase_text = " ".join(phrase)
    best = 0.0
    for size in range(low, high + 1):
        for i in range(len(title_tokens) - size + 1):
            candidate = " ".join(title_tokens[i : i + size])
            best = max(best, SequenceMatcher(None, phrase_text, candidate).ratio())
    return best


def has_opposed_ko_o_prefix(title_tokens: tuple[str, ...], tech_tokens: tuple[str, ...]) -> bool:
    if len(tech_tokens) < 3 or tech_tokens[0] not in {"ko", "o"}:
        return False
    opposite = "o" if tech_tokens[0] == "ko" else "ko"
    return contains_phrase(title_tokens, (opposite, *tech_tokens[1:]))


def requires_adjacent_prefix_match(tokens: tuple[str, ...]) -> bool:
    return len(tokens) >= 2 and tokens[0] in {"ko", "o"}


def load_techniques(dataset: Path) -> list[Technique]:
    techniques = []
    seen = set()
    folder_names = {path.name for path in dataset.iterdir() if path.is_dir()}
    for path in sorted(dataset.iterdir()):
        if not path.is_dir():
            continue
        if path.name.startswith(".") or path.name.startswith("_"):
            continue
        if path.name.startswith("youtube_playlist_"):
            continue
        canonical = canonical_folder(path.name)
        if canonical != path.name and (canonical in folder_names or canonical in KNOWN_THROW_FOLDERS):
            continue
        phrase = normalize_text(path.name.replace("-", " "))
        toks = tokens_for(path.name.replace("-", " "))
        if toks:
            techniques.append(Technique(path.name, phrase, toks))
            seen.add(path.name)
    for folder in sorted(KNOWN_THROW_FOLDERS):
        if folder in seen:
            continue
        toks = tokens_for(folder.replace("-", " "))
        if toks:
            techniques.append(Technique(folder, normalize_text(folder.replace("-", " ")), toks))
            seen.add(folder)
    return techniques


def score_technique(title: str, technique: Technique) -> Match | None:
    title_tokens = useful_title_tokens(title)
    if not title_tokens:
        return None
    tech_tokens = technique.tokens
    if contains_phrase(title_tokens, tech_tokens):
        return Match(technique.folder, 0.97 + min(len(tech_tokens), 6) * 0.01, "phrase", tech_tokens)
    if requires_adjacent_prefix_match(tech_tokens):
        for alias in TECHNIQUE_ALIASES.get(technique.folder, ()):
            blocked = ALIAS_BLOCKED_TOKENS.get((technique.folder, alias), set())
            if not blocked & set(title_tokens) and contains_phrase(title_tokens, alias):
                return Match(technique.folder, 0.89 + min(len(alias), 4) * 0.02, "alias", alias)
        return None
    if len(tech_tokens) >= 2 and is_subsequence(tech_tokens, title_tokens):
        return Match(technique.folder, 0.90 + min(len(tech_tokens), 6) * 0.01, "ordered_tokens", tech_tokens)
    if len(tech_tokens) >= 2 and set(tech_tokens).issubset(title_tokens):
        return Match(technique.folder, 0.84 + min(len(tech_tokens), 6) * 0.01, "all_tokens", tech_tokens)
    title_token_set = set(title_tokens)
    for alias in TECHNIQUE_ALIASES.get(technique.folder, ()):
        blocked = ALIAS_BLOCKED_TOKENS.get((technique.folder, alias), set())
        if blocked & title_token_set:
            continue
        if contains_phrase(title_tokens, alias) or (alias[0] not in {"ko", "o"} and set(alias).issubset(title_token_set)):
            return Match(technique.folder, 0.89 + min(len(alias), 4) * 0.02, "alias", alias)
    if has_opposed_ko_o_prefix(title_tokens, tech_tokens):
        return None
    ratio = best_window_ratio(title_tokens, tech_tokens)
    if ratio >= FUZZY_WINDOW_MIN_RATIO and len(tech_tokens) >= 2:
        return Match(technique.folder, ratio, "fuzzy_window", tech_tokens)
    return None


def prune_generic_matches(matches: Iterable[Match]) -> tuple[Match, ...]:
    ranked = sorted(matches, key=lambda m: (m.score, len(m.tokens)), reverse=True)
    kept: list[Match] = []
    for match in ranked:
        match_set = set(match.tokens)
        if any(match_set < set(existing.tokens) for existing in kept):
            continue
        kept.append(match)
    return tuple(sorted(kept, key=lambda m: (m.folder, -m.score)))


def match_title(title: str, techniques: list[Technique], threshold: float) -> tuple[Match, ...]:
    matches = [match for technique in techniques if (match := score_technique(title, technique))]
    matches = [match for match in matches if match.score >= threshold]
    return canonicalize_matches(prune_generic_matches(matches))


def canonicalize_matches(matches: Iterable[Match]) -> tuple[Match, ...]:
    by_folder: dict[str, Match] = {}
    for match in matches:
        folder = canonical_folder(match.folder)
        reason = match.reason
        if folder != match.folder:
            reason = f"{reason};canonical:{match.folder}"
        canonical = Match(folder, match.score, reason, match.tokens)
        existing = by_folder.get(folder)
        if existing is None or (canonical.score, len(canonical.tokens)) > (existing.score, len(existing.tokens)):
            by_folder[folder] = canonical
    return tuple(sorted(by_folder.values(), key=lambda m: (m.folder, -m.score)))


def unique_urls(urls: Iterable[str]) -> list[str]:
    seen = set()
    result = []
    for url in urls:
        if url not in seen:
            result.append(url)
            seen.add(url)
    return result


def extract_playlist_id(url: str) -> str:
    match = re.search(r"[?&]list=([^&]+)", url)
    return match.group(1) if match else "unknown"


def load_archive_ids(archive: Path) -> set[str]:
    if not archive.exists():
        return set()
    ids = set()
    for line in archive.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[0] == "youtube":
            ids.add(parts[1])
    return ids


def extract_plan(
    urls: list[str],
    techniques: list[Technique],
    threshold: float,
    archive: Path,
) -> tuple[list[PlannedVideo], list[dict[str, str]]]:
    archive_ids = load_archive_ids(archive)
    ydl_opts = {
        "extract_flat": "in_playlist",
        "ignoreerrors": True,
        "quiet": True,
        "skip_download": True,
    }
    planned: list[PlannedVideo] = []
    rows: list[dict[str, str]] = []
    seen_ids = set()
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        for playlist_url in urls:
            fallback_id = extract_playlist_id(playlist_url)
            print(f"metadata: {playlist_url}", flush=True)
            try:
                info = ydl.extract_info(playlist_url, download=False)
            except Exception as exc:
                rows.append(
                    {
                        "playlist_id": fallback_id,
                        "playlist_title": "",
                        "video_id": "",
                        "title": "",
                        "url": playlist_url,
                        "matched_folders": "",
                        "scores": "",
                        "reasons": f"metadata_error:{exc}",
                        "already_archived": "",
                    }
                )
                continue
            if not info:
                continue
            playlist_id = info.get("id") or fallback_id
            playlist_title = info.get("title") or ""
            entries = info.get("entries") or []
            for entry in entries:
                if not entry:
                    continue
                video_id = entry.get("id") or entry.get("url") or ""
                title = entry.get("title") or ""
                if not video_id or not title:
                    continue
                matches = match_title(title, techniques, threshold)
                already_archived = video_id in archive_ids
                rows.append(
                    {
                        "playlist_id": playlist_id,
                        "playlist_title": playlist_title,
                        "video_id": video_id,
                        "title": title,
                        "url": f"https://www.youtube.com/watch?v={video_id}",
                        "matched_folders": "|".join(match.folder for match in matches),
                        "scores": "|".join(f"{match.score:.3f}" for match in matches),
                        "reasons": "|".join(match.reason for match in matches),
                        "already_archived": str(already_archived),
                    }
                )
                if not matches or video_id in seen_ids:
                    continue
                seen_ids.add(video_id)
                planned.append(
                    PlannedVideo(
                        playlist_id=playlist_id,
                        playlist_title=playlist_title,
                        video_id=video_id,
                        title=title,
                        url=f"https://www.youtube.com/watch?v={video_id}",
                        matches=matches,
                    )
                )
    return planned, rows


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "playlist_id",
        "playlist_title",
        "video_id",
        "title",
        "url",
        "matched_folders",
        "scores",
        "reasons",
        "already_archived",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def ensure_free_space(dataset: Path, min_free_gib: float) -> None:
    free = shutil.disk_usage(dataset).free / (1024**3)
    if free < min_free_gib:
        raise RuntimeError(f"free disk space {free:.1f} GiB is below {min_free_gib:.1f} GiB")


def js_runtime_option() -> dict[str, dict[str, str]]:
    for runtime in ("node", "deno", "bun"):
        path = shutil.which(runtime)
        if path:
            return {runtime: {"path": path}}
    return {"deno": {}}


def parse_cookies_from_browser(value: str | None) -> tuple[str, str | None, str | None, str | None] | None:
    if not value:
        return None
    browser, _, profile = value.partition(":")
    browser = browser.strip().lower()
    profile = profile.strip() or None
    return (browser, profile, None, None)


def ydl_options(
    output_dir: Path,
    archive: Path,
    cookies_from_browser: str | None,
    sleep_requests: float,
    sleep_interval: float,
    max_sleep_interval: float,
) -> dict[str, object]:
    return {
        "cookiesfrombrowser": parse_cookies_from_browser(cookies_from_browser),
        "download_archive": str(archive),
        "format": "bv*[height<=720]+ba/b[height<=720]/best[height<=720]/best",
        "ignoreerrors": True,
        "js_runtimes": js_runtime_option(),
        "max_sleep_interval": max_sleep_interval,
        "nooverwrites": True,
        "outtmpl": str(output_dir / "%(title)s [%(id)s].%(ext)s"),
        "quiet": False,
        "remote_components": ["ejs:github"],
        "retries": 3,
        "sleep_interval": sleep_interval,
        "sleep_interval_requests": sleep_requests,
        "yes_playlist": False,
    }


def download_planned(
    planned: list[PlannedVideo],
    dataset: Path,
    archive: Path,
    min_free_gib: float,
    cookies_from_browser: str | None,
    sleep_requests: float,
    sleep_interval: float,
    max_sleep_interval: float,
) -> None:
    by_playlist: dict[str, list[PlannedVideo]] = {}
    for video in planned:
        by_playlist.setdefault(video.playlist_id, []).append(video)
    for playlist_id, videos in by_playlist.items():
        ensure_free_space(dataset, min_free_gib)
        output_dir = dataset / f"youtube_playlist_{playlist_id}"
        output_dir.mkdir(parents=True, exist_ok=True)
        urls = [video.url for video in videos]
        print(f"download: {playlist_id} matched={len(urls)}", flush=True)
        if cookies_from_browser:
            for index, url in enumerate(urls, start=1):
                ensure_free_space(dataset, min_free_gib)
                print(f"download: {playlist_id} video={index}/{len(urls)}", flush=True)
                with yt_dlp.YoutubeDL(
                    ydl_options(
                        output_dir,
                        archive,
                        cookies_from_browser,
                        sleep_requests,
                        sleep_interval,
                        max_sleep_interval,
                    )
                ) as ydl:
                    ydl.download([url])
        else:
            with yt_dlp.YoutubeDL(
                ydl_options(
                    output_dir,
                    archive,
                    cookies_from_browser,
                    sleep_requests,
                    sleep_interval,
                    max_sleep_interval,
                )
            ) as ydl:
                ydl.download(urls)


def consolidate_alias_folders(dataset: Path, techniques: list[Technique], threshold: float) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for alias, canonical in sorted(CANONICAL_FOLDER_ALIASES.items()):
        source_dir = dataset / alias
        if not source_dir.is_dir():
            continue
        for source in sorted(source_dir.iterdir()):
            if not source.is_file() or source.suffix.lower() not in MEDIA_EXTENSIONS:
                continue
            title = re.sub(r"\s+\[[A-Za-z0-9_-]{6,}\]$", "", source.stem)
            matches = match_title(title, techniques, threshold)
            if not matches:
                matches = (Match(canonical, 1.0, f"canonical_alias:{alias}", tokens_for(canonical)),)
            for match in matches:
                target_dir = dataset / match.folder
                target_dir.mkdir(parents=True, exist_ok=True)
                target = unique_target_path(target_dir, source, source_dir.name)
                action = link_or_skip(source, target)
                reason = match.reason
                if not reason.startswith("canonical_alias:"):
                    reason = f"{reason};source_alias:{alias}"
                rows.append(
                    {
                        "source": str(source),
                        "title": title,
                        "target": str(target),
                        "target_folder": match.folder,
                        "score": f"{match.score:.3f}",
                        "reason": reason,
                        "action": action,
                    }
                )
    return rows


def classify_sources(dataset: Path, techniques: list[Technique], threshold: float, manifest: Path) -> None:
    rows: list[dict[str, str]] = []
    for source_dir in sorted(dataset.glob("youtube_playlist_*")):
        if not source_dir.is_dir():
            continue
        for source in sorted(source_dir.iterdir()):
            if not source.is_file() or source.suffix.lower() not in MEDIA_EXTENSIONS:
                continue
            title = re.sub(r"\s+\[[A-Za-z0-9_-]{6,}\]$", "", source.stem)
            matches = match_title(title, techniques, threshold)
            if not matches:
                rows.append(
                    {
                        "source": str(source),
                        "title": title,
                        "target": "",
                        "target_folder": "",
                        "score": "",
                        "reason": "unmatched",
                        "action": "left_in_source",
                    }
                )
                continue
            for match in matches:
                target_dir = dataset / match.folder
                target_dir.mkdir(parents=True, exist_ok=True)
                target = unique_target_path(target_dir, source, source_dir.name)
                action = link_or_skip(source, target)
                rows.append(
                    {
                        "source": str(source),
                        "title": title,
                        "target": str(target),
                        "target_folder": match.folder,
                        "score": f"{match.score:.3f}",
                        "reason": match.reason,
                        "action": action,
                    }
                )
    rows.extend(consolidate_alias_folders(dataset, techniques, threshold))
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = ["source", "title", "target", "target_folder", "score", "reason", "action"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    linked = sum(1 for row in rows if row["action"] in {"linked", "exists_same_file", "exists_same_size"})
    unmatched = sum(1 for row in rows if row["action"] == "left_in_source")
    print(f"classified_rows={len(rows)} linked_or_existing={linked} unmatched={unmatched}", flush=True)


def unique_target_path(target_dir: Path, source: Path, source_tag: str) -> Path:
    def is_same_existing(path: Path) -> bool:
        try:
            return path.exists() and os.path.samefile(source, path)
        except OSError:
            return False

    target = target_dir / source.name
    if not target.exists():
        return target
    if is_same_existing(target):
        return target
    stem = target.stem
    suffix = target.suffix
    tagged = target_dir / f"{stem} [{source_tag}]{suffix}"
    if not tagged.exists():
        return tagged
    if is_same_existing(tagged):
        return tagged
    counter = 2
    while True:
        candidate = target_dir / f"{stem} [{source_tag} {counter}]{suffix}"
        if not candidate.exists():
            return candidate
        if is_same_existing(candidate):
            return candidate
        counter += 1


def link_or_skip(source: Path, target: Path) -> str:
    if target.exists():
        try:
            if os.path.samefile(source, target):
                return "exists_same_file"
        except OSError:
            pass
        if source.stat().st_size == target.stat().st_size:
            return "exists_same_size"
        return "exists_collision"
    try:
        os.link(source, target)
        return "linked"
    except OSError:
        shutil.copy2(source, target)
        return "copied"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("dataset"))
    parser.add_argument("--archive", type=Path, default=Path("dataset/youtube_playlists_new_archive.txt"))
    parser.add_argument("--plan-csv", type=Path, default=Path("dataset/youtube_playlists_semantic_plan.csv"))
    parser.add_argument("--manifest-csv", type=Path, default=Path("dataset/youtube_playlists_classification_manifest.csv"))
    parser.add_argument("--threshold", type=float, default=0.86)
    parser.add_argument("--min-free-gib", type=float, default=30.0)
    parser.add_argument("--cookies-from-browser", help="Browser name, optionally with profile, e.g. chrome or chrome:Profile 1")
    parser.add_argument("--sleep-requests", type=float, default=0.0)
    parser.add_argument("--sleep-interval", type=float, default=0.0)
    parser.add_argument("--max-sleep-interval", type=float, default=0.0)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--classify-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dataset = args.dataset
    dataset.mkdir(parents=True, exist_ok=True)
    techniques = load_techniques(dataset)
    if not techniques:
        print("No technique folders found.", file=sys.stderr)
        return 2
    print(f"technique_folders={len(techniques)}", flush=True)
    if args.classify_only:
        classify_sources(dataset, techniques, args.threshold, args.manifest_csv)
        return 0
    urls = unique_urls(PLAYLIST_URLS)
    planned, rows = extract_plan(urls, techniques, args.threshold, args.archive)
    write_csv(args.plan_csv, rows)
    matched_rows = sum(1 for row in rows if row["matched_folders"])
    archived_matched = sum(1 for row in rows if row["matched_folders"] and row["already_archived"] == "True")
    print(
        f"plan_rows={len(rows)} matched_rows={matched_rows} unique_matched_videos={len(planned)} "
        f"matched_already_archived={archived_matched}",
        flush=True,
    )
    if args.plan_only:
        return 0
    download_planned(
        planned,
        dataset,
        args.archive,
        args.min_free_gib,
        args.cookies_from_browser,
        args.sleep_requests,
        args.sleep_interval,
        args.max_sleep_interval,
    )
    classify_sources(dataset, techniques, args.threshold, args.manifest_csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
