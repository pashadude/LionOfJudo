#!/usr/bin/env python3
"""Local video-only coach review commands."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipeline.video_review_import import import_session


def _blue_seed(value: str) -> tuple[float, float, float, float]:
    parts = value.split(",")
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("blue seed mora imati oblik x1,y1,x2,y2")
    try:
        return tuple(float(part) for part in parts)  # type: ignore[return-value]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("blue seed mora sadrzati brojeve") from exc


def _import_command(args: argparse.Namespace) -> int:
    with Path(args.anchors_json).open(encoding="utf-8") as handle:
        anchor_payload = json.load(handle)
    anchors = anchor_payload.get("anchors", anchor_payload) if isinstance(anchor_payload, dict) else anchor_payload
    review_path = import_session(
        sony=args.sony,
        iphone=args.iphone,
        output_dir=args.session_dir,
        anchors=anchors,
        injury_cutoff_s=args.injury_cutoff_s,
        blue_seed=args.blue_seed_sony,
        transcript_path=args.transcript_json,
        force_reimport=args.force_reimport,
    )
    print(f"Uvoz zavrsen: {review_path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    import_parser = subparsers.add_parser(
        "import", help="uvezi Sony/iPhone sesiju za lokalni pregled"
    )
    import_parser.add_argument(
        "--sony", required=True, type=Path,
        help="apsolutna putanja do Sony videa (glavna vremenska osa)",
    )
    import_parser.add_argument(
        "--iphone", required=True, type=Path,
        help="apsolutna putanja do iPhone videa",
    )
    import_parser.add_argument(
        "--session-dir", required=True, type=Path,
        help="direktorijum sesije za lokalne izlaze",
    )
    import_parser.add_argument(
        "--anchors-json", required=True, type=Path,
        help="JSON sa dva rucno potvrdjena trostruka dodira",
    )
    import_parser.add_argument(
        "--injury-cutoff-sony-s", required=True, type=float,
        help="potvrdjeni kraj normalne obrade na Sony osi u sekundama",
    )
    import_parser.add_argument(
        "--blue-seed-sony", required=True, type=_blue_seed,
        help="pocetni okvir plavog sportiste: x1,y1,x2,y2",
    )
    import_parser.add_argument(
        "--transcript-json", type=Path, default=None,
        help="opcionalni lokalni Whisper JSON",
    )
    import_parser.add_argument(
        "--force-reimport", action="store_true",
        help="ponovi uvoz uz ocuvanje postojecih trenerovih zabelezbi",
    )
    import_parser.set_defaults(handler=_import_command)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
