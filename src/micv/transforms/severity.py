from __future__ import annotations

import random


def normalize_severity(severity: str) -> str:
    severity_aliases = {
        "easy": "train",
        "medium": "val",
        "none": "none",
        "off": "none",
        "disabled": "none",
    }
    return severity_aliases.get(severity, severity)


def sample_effective_severity(severity: str) -> str:
    normalized_severity = normalize_severity(severity)
    if normalized_severity == "mixed":
        return random.choices(["train", "val", "hard"], weights=[0.4, 0.35, 0.25], k=1)[0]
    if normalized_severity == "test":
        return "hard"
    return normalized_severity