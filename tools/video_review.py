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
from pipeline.video_review_migration import migrate_session
from coach_app.server import create_server


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
        analysis_fps=args.analysis_fps,
        model_path=args.model_path,
        device=args.device,
        event_threshold=args.event_threshold,
    )
    print(f"Uvoz zavrsen: {review_path}")
    return 0


def _serve_command(args: argparse.Namespace) -> int:
    server = create_server(args.session_dir, port=args.port)
    print(f"Pregled dostupan na {server.base_url}", flush=True)
    try:
        server.httpd.serve_forever()
    except KeyboardInterrupt:
        return 0
    finally:
        server.shutdown()
    return 0


def _migrate_command(args: argparse.Namespace) -> int:
    review_path = migrate_session(args.session_dir)
    print(f"Migracija završena: {review_path}")
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
        dest="injury_cutoff_s",
        help="potvrdjeni kraj normalne obrade na Sony osi u sekundama",
    )
    import_parser.add_argument(
        "--blue-seed-sony", required=True, type=_blue_seed,
        help="pocetni okvir plavog sportiste: x1,y1,x2,y2",
    )
    import_parser.add_argument(
        "--analysis-fps", type=float, default=None,
        help="opcionalna stopa analize; npr. 3.0 za grubo uzorkovanje",
    )
    import_parser.add_argument(
        "--model-path", type=Path, default=Path("yolo11x-pose.pt"),
        help="postojeca lokalna YOLO pose tezina; uvoz ne preuzima model",
    )
    import_parser.add_argument(
        "--device", default="mps",
        help="uredjaj za lokalni YOLO (npr. mps, cpu ili cuda:0)",
    )
    import_parser.add_argument(
        "--event-threshold", type=float, default=0.5,
        help="prag normalizovanog video-pokreta (podrazumevano 0.5)",
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

    migrate_parser = subparsers.add_parser(
        "migrate",
        help="migriraj postojeću sesiju bez ponovne video-analize",
    )
    migrate_parser.add_argument(
        "--session-dir",
        required=True,
        type=Path,
        help="direktorijum postojeće sesije sa review.json",
    )
    migrate_parser.set_defaults(handler=_migrate_command)

    serve_parser = subparsers.add_parser(
        "serve", help="pokreni lokalni pregled u pregledaču"
    )
    serve_parser.add_argument(
        "--session-dir", required=True, type=Path,
        help="direktorijum uvožene sesije sa review.json",
    )
    serve_parser.add_argument(
        "--port", type=int, default=8765,
        help="lokalni port (podrazumevano 8765)",
    )
    serve_parser.set_defaults(handler=_serve_command)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
