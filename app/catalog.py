from __future__ import annotations

import json
import logging
from pathlib import Path
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


def _clean(text: str) -> str:
    """Collapse embedded newlines/tabs into single spaces."""
    return " ".join(text.split())

KEY_TO_CODE: dict[str, str] = {
    "Knowledge & Skills": "K",
    "Personality & Behavior": "P",
    "Ability & Aptitude": "A",
    "Simulations": "S",
    "Assessment Exercises": "S",
    "Biodata & Situational Judgment": "B",
    "Competencies": "C",
    "Development & 360": "D",
}


@dataclass
class Assessment:
    entity_id: str
    name: str
    link: str
    description: str
    keys: list[str]
    job_levels: list[str]
    duration: str
    remote: str
    adaptive: str
    languages: list[str]
    type_codes: str = ""
    rich_text: str = ""

    def __post_init__(self) -> None:
        codes = sorted({KEY_TO_CODE.get(k, "") for k in self.keys} - {""})
        self.type_codes = ",".join(codes)
        self.rich_text = (
            f"{self.name}. {self.description} "
            f"Types: {', '.join(self.keys)}. "
            f"Levels: {', '.join(self.job_levels)}. "
            f"Duration: {self.duration or 'Not specified'}. "
            f"Remote: {self.remote}."
        )


@dataclass
class Catalog:
    assessments: list[Assessment] = field(default_factory=list)
    by_name: dict[str, Assessment] = field(default_factory=dict)
    by_url: dict[str, Assessment] = field(default_factory=dict)
    by_id: dict[str, Assessment] = field(default_factory=dict)

    @classmethod
    def load(cls, path: str | Path | None = None) -> Catalog:
        if path is None:
            path = Path(__file__).resolve().parent.parent / "data" / "shl_product_catalog.json"
        path = Path(path)
        decoder = json.JSONDecoder(strict=False)
        raw = decoder.decode(path.read_text(encoding="utf-8"))
        catalog = cls()
        for item in raw:
            a = Assessment(
                entity_id=item.get("entity_id", ""),
                name=_clean(item.get("name", "")),
                link=item.get("link", ""),
                description=_clean(item.get("description", "")),
                keys=item.get("keys", []),
                job_levels=item.get("job_levels", []),
                duration=item.get("duration", ""),
                remote=item.get("remote", ""),
                adaptive=item.get("adaptive", ""),
                languages=item.get("languages", []),
            )
            catalog.assessments.append(a)
            catalog.by_name[a.name.lower()] = a
            catalog.by_url[a.link] = a
            catalog.by_id[a.entity_id] = a
        logger.info("Loaded %d assessments from catalog", len(catalog.assessments))
        return catalog
