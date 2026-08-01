"""Atomic Serbian-Latin CSV and Markdown reports for the review contract."""

from __future__ import annotations

import csv
from html import escape
import json
import os
from pathlib import Path
from typing import Any, Mapping

from pipeline.trainer_ai_state import active_ai_evaluation


REPORT_FIELDS = (
    "event_id",
    "Vreme početka (s)",
    "Vreme kraja (s)",
    "Predlog tehnike",
    "Potvrđena tehnika",
    "Glasovna fraza",
    "Početak glasovne fraze (s)",
    "Kraj glasovne fraze (s)",
    "Pouzdanost glasa",
    "Brzina ulaska norm",
    "Rotacija trupa 2D (step/s)",
    "Promena visine kukova norm",
    "Širina stava norm",
    "Vreme oporavka (s)",
    "Intenzitet pokreta (0-100)",
    "Ocena",
    "Napomena",
    "Event revizija",
    "Analysis fingerprint",
    "AI evaluator",
    "AI status",
    "AI ocena",
    "AI pouzdanost",
    "AI otkriven u",
    "AI razlog",
    "AI dokazi (JSON)",
    "IMU eksperimentalno (JSON)",
    "Trener pre-AI revizija",
    "Trener pre-AI status vidljivosti",
    "Trener pre-AI tehnika",
    "Trener pre-AI ocena",
    "Trener pre-AI razlog",
    "Trener pre-AI Sony trenuci (JSON)",
    "Trener pre-AI zaključano u",
    "Trener procene (JSON)",
    "Odnos trenera prema AI",
    "Razlog odnosa prema AI",
    "Procene AI dokaza (JSON)",
    "Procene AI predloga (JSON)",
    "Odgovor sačuvan u",
    "Status",
    "Isključeno iz statistike",
)


def event_is_injury(event: Mapping[str, Any]) -> bool:
    return bool(
        event.get("prijavljen_povredni_dogadjaj")
        or event.get("iskljuceno_iz_statistike")
        or event.get("status") == "povreda"
    )


def _display(value: Any) -> Any:
    return "" if value is None else value


def _compact_json(value: Any) -> str:
    if value is None:
        return ""
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _current_pre_ai(event: Mapping[str, Any]) -> Mapping[str, Any] | None:
    matches = [
        row
        for row in event.get("trener_procene", [])
        if isinstance(row, Mapping)
        and row.get("faza") == "pre_ai"
        and row.get("event_revision") == event.get("event_revision")
        and row.get("analysis_fingerprint") == event.get("analysis_fingerprint")
    ]
    return matches[0] if len(matches) == 1 else None


def _active_feedback(event: Mapping[str, Any]) -> Mapping[str, Any] | None:
    duel = event.get("aktivni_duel")
    if not isinstance(duel, Mapping):
        return None
    reference = (
        duel.get("event_revision"),
        duel.get("analysis_fingerprint"),
        duel.get("trener_revizija"),
        duel.get("evaluator_id"),
    )
    matches = [
        row
        for row in event.get("procene_ai_predloga", [])
        if isinstance(row, Mapping)
        and (
            row.get("event_revision"),
            row.get("analysis_fingerprint"),
            row.get("trener_revizija"),
            row.get("evaluator_id"),
        )
        == reference
    ]
    return matches[0] if len(matches) == 1 else None


def report_row(
    event: Mapping[str, Any], *, include_unrevealed: bool = False
) -> dict[str, Any]:
    status_parts = []
    injury = event_is_injury(event)
    if injury:
        status_parts.append("Prijavljen povredni događaj")
    if str(event.get("vidljivost", "")).lower() in {
        "nedovoljno vidljivo",
        "nedovoljno_vidljivo",
    }:
        status_parts.append("Nedovoljno vidljivo")
    pre_ai = None if injury else _current_pre_ai(event)
    ai = None if injury else active_ai_evaluation(event)
    ai_visible = ai is not None and (
        include_unrevealed or ai.get("ai_otkriven_u") is not None
    )
    visible_ai = ai if ai_visible else None
    feedback = _active_feedback(event) if ai_visible else None
    trainer_rows = None if injury else event.get("trener_procene", [])
    feedback_rows = (
        event.get("procene_ai_predloga", []) if ai_visible and not injury else None
    )
    row = {
        "event_id": event.get("event_id", ""),
        "Vreme početka (s)": _display(event.get("sony_start_s")),
        "Vreme kraja (s)": _display(event.get("sony_end_s")),
        "Predlog tehnike": _display(event.get("predlog_tehnike")),
        "Potvrđena tehnika": "" if injury else _display(event.get("potvrdena_tehnika")),
        "Glasovna fraza": _display(event.get("glasovna_fraza")),
        "Početak glasovne fraze (s)": _display(event.get("glasovna_fraza_pocetak_s")),
        "Kraj glasovne fraze (s)": _display(event.get("glasovna_fraza_kraj_s")),
        "Pouzdanost glasa": _display(event.get("pouzdanost_glasa")),
        "Brzina ulaska norm": _display(event.get("brzina_ulaska_norm")),
        "Rotacija trupa 2D (step/s)": _display(event.get("rotacija_trupa_2d_dps")),
        "Promena visine kukova norm": _display(event.get("promena_visine_kukova_norm")),
        "Širina stava norm": _display(event.get("sirina_stava_norm")),
        "Vreme oporavka (s)": _display(event.get("vreme_oporavka_s")),
        "Intenzitet pokreta (0-100)": _display(event.get("intenzitet_pokreta_0_100")),
        "Ocena": "" if injury else _display(event.get("ocena")),
        "Napomena": "" if injury else _display(event.get("napomena")),
        "Event revizija": "" if injury else _display(event.get("event_revision")),
        "Analysis fingerprint": "" if injury else _display(event.get("analysis_fingerprint")),
        "AI evaluator": _display(visible_ai.get("evaluator_id")) if visible_ai else "",
        "AI status": _display(visible_ai.get("status")) if visible_ai else "",
        "AI ocena": _display(visible_ai.get("predlozena_ocena")) if visible_ai else "",
        "AI pouzdanost": _display(visible_ai.get("pouzdanost_0_1")) if visible_ai else "",
        "AI otkriven u": _display(visible_ai.get("ai_otkriven_u")) if visible_ai else "",
        "AI razlog": _display(visible_ai.get("razlog")) if visible_ai else "",
        "AI dokazi (JSON)": _compact_json(visible_ai.get("dokazi")) if visible_ai else "",
        "IMU eksperimentalno (JSON)": (
            _compact_json(event.get("imu_eksperimentalno")) if ai_visible else ""
        ),
        "Trener pre-AI revizija": _display(pre_ai.get("revizija")) if pre_ai else "",
        "Trener pre-AI status vidljivosti": (
            _display(pre_ai.get("status_vidljivosti")) if pre_ai else ""
        ),
        "Trener pre-AI tehnika": _display(pre_ai.get("potvrdena_tehnika")) if pre_ai else "",
        "Trener pre-AI ocena": _display(pre_ai.get("ocena")) if pre_ai else "",
        "Trener pre-AI razlog": _display(pre_ai.get("razlog")) if pre_ai else "",
        "Trener pre-AI Sony trenuci (JSON)": (
            _compact_json(pre_ai.get("citirani_sony_trenuci_s")) if pre_ai else ""
        ),
        "Trener pre-AI zaključano u": _display(pre_ai.get("zakljucano_u")) if pre_ai else "",
        "Trener procene (JSON)": _compact_json(trainer_rows),
        "Odnos trenera prema AI": _display(feedback.get("odnos")) if feedback else "",
        "Razlog odnosa prema AI": _display(feedback.get("razlog")) if feedback else "",
        "Procene AI dokaza (JSON)": (
            _compact_json(feedback.get("procene_dokaza")) if feedback else ""
        ),
        "Procene AI predloga (JSON)": _compact_json(feedback_rows),
        "Odgovor sačuvan u": _display(feedback.get("sacuvano_u")) if feedback else "",
        "Status": "; ".join(status_parts) or event.get("status", ""),
        "Isključeno iz statistike": "da" if injury else "ne",
    }
    return row


def report_rows(
    review: Mapping[str, Any], *, include_unrevealed: bool = False
) -> list[dict[str, Any]]:
    events = review.get("events", [])
    if not isinstance(events, list):
        raise ValueError("events mora biti JSON lista")
    return [
        report_row(event, include_unrevealed=include_unrevealed)
        for event in events
        if isinstance(event, Mapping)
    ]


def markdown_cell(value: object) -> str:
    """Render one safe table cell with normalized line endings."""
    text = str(value if value is not None else "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = escape(text, quote=False)
    text = text.replace("\\", "\\\\").replace("|", "\\|")
    return text.replace("\n", "<br>")


def write_reports(
    review_path: Path,
    review: Mapping[str, Any],
    *,
    include_unrevealed: bool = False,
) -> None:
    events = review.get("events", [])
    if not isinstance(events, list):
        raise ValueError("events mora biti JSON lista")
    rows = report_rows(review, include_unrevealed=include_unrevealed)
    review_path = Path(review_path)

    csv_path = review_path.with_name("izvestaj.csv")
    csv_tmp = csv_path.with_name("izvestaj.csv.tmp")
    try:
        with csv_tmp.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=REPORT_FIELDS, extrasaction="raise")
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(csv_tmp, csv_path)
    finally:
        csv_tmp.unlink(missing_ok=True)

    markdown_path = review_path.with_name("izvestaj.md")
    markdown_tmp = markdown_path.with_name("izvestaj.md.tmp")
    try:
        with markdown_tmp.open("w", encoding="utf-8", newline="\n") as handle:
            normal_count = sum(
                1 for event in events
                if isinstance(event, Mapping) and not event_is_injury(event)
            )
            handle.write("# Izveštaj trenerskog pregleda\n\n")
            handle.write(f"Normalni događaji u statistici: {normal_count}\n\n")
            handle.write("| " + " | ".join(REPORT_FIELDS) + " |\n")
            handle.write("| " + " | ".join("---" for _ in REPORT_FIELDS) + " |\n")
            for row in rows:
                handle.write(
                    "| "
                    + " | ".join(markdown_cell(row[field]) for field in REPORT_FIELDS)
                    + " |\n"
                )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(markdown_tmp, markdown_path)
    finally:
        markdown_tmp.unlink(missing_ok=True)


__all__ = [
    "REPORT_FIELDS",
    "event_is_injury",
    "markdown_cell",
    "report_row",
    "report_rows",
    "write_reports",
]
