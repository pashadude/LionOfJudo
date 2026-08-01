"""Atomic Serbian-Latin CSV and Markdown reports for the review contract."""

from __future__ import annotations

import csv
from html import escape
import os
from pathlib import Path
from typing import Any, Mapping


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
    "Status",
    "Isključeno iz statistike",
)


def event_is_injury(event: Mapping[str, Any]) -> bool:
    return bool(
        event.get("prijavljen_povredni_dogadjaj")
        or event.get("iskljuceno_iz_statistike")
        or event.get("status") == "povreda"
    )


def report_row(event: Mapping[str, Any]) -> dict[str, Any]:
    status_parts = []
    if event_is_injury(event):
        status_parts.append("Prijavljen povredni događaj")
    if str(event.get("vidljivost", "")).lower() in {
        "nedovoljno vidljivo",
        "nedovoljno_vidljivo",
    }:
        status_parts.append("Nedovoljno vidljivo")
    return {
        "event_id": event.get("event_id", ""),
        "Vreme početka (s)": event.get("sony_start_s", ""),
        "Vreme kraja (s)": event.get("sony_end_s", ""),
        "Predlog tehnike": event.get("predlog_tehnike", ""),
        "Potvrđena tehnika": event.get("potvrdena_tehnika", ""),
        "Glasovna fraza": event.get("glasovna_fraza", ""),
        "Početak glasovne fraze (s)": event.get("glasovna_fraza_pocetak_s", ""),
        "Kraj glasovne fraze (s)": event.get("glasovna_fraza_kraj_s", ""),
        "Pouzdanost glasa": event.get("pouzdanost_glasa", ""),
        "Brzina ulaska norm": event.get("brzina_ulaska_norm", ""),
        "Rotacija trupa 2D (step/s)": event.get("rotacija_trupa_2d_dps", ""),
        "Promena visine kukova norm": event.get("promena_visine_kukova_norm", ""),
        "Širina stava norm": event.get("sirina_stava_norm", ""),
        "Vreme oporavka (s)": event.get("vreme_oporavka_s", ""),
        "Intenzitet pokreta (0-100)": event.get("intenzitet_pokreta_0_100", ""),
        "Ocena": event.get("ocena", ""),
        "Napomena": event.get("napomena", ""),
        "Status": "; ".join(status_parts) or event.get("status", ""),
        "Isključeno iz statistike": "da" if event_is_injury(event) else "ne",
    }


def markdown_cell(value: object) -> str:
    """Render one safe table cell with normalized line endings."""
    text = str(value if value is not None else "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = escape(text, quote=False)
    text = text.replace("\\", "\\\\").replace("|", "\\|")
    return text.replace("\n", "<br>")


def write_reports(review_path: Path, review: Mapping[str, Any]) -> None:
    events = review.get("events", [])
    if not isinstance(events, list):
        raise ValueError("events mora biti JSON lista")
    rows = [report_row(event) for event in events if isinstance(event, Mapping)]
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
    "write_reports",
]
