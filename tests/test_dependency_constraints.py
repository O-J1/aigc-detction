from __future__ import annotations

import re
import tomllib
from pathlib import Path


_EXACT_REQUIREMENT = re.compile(r"^(?P<name>[A-Za-z0-9_.-]+)==(?P<version>[^;]+)$")


def _normalise(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _project_requirements() -> dict[str, str]:
    project = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))["project"]
    requirements = list(project["dependencies"])
    for values in project.get("optional-dependencies", {}).values():
        requirements.extend(values)

    parsed: dict[str, str] = {}
    for requirement in requirements:
        match = _EXACT_REQUIREMENT.fullmatch(requirement)
        assert match is not None, f"Dependency is not exactly pinned: {requirement}"
        parsed[_normalise(match.group("name"))] = match.group("version")
    return parsed


def _constraint_requirements() -> dict[str, str]:
    parsed: dict[str, str] = {}
    for raw_line in Path("constraints/py313-cu130.txt").read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        match = _EXACT_REQUIREMENT.fullmatch(line)
        assert match is not None, f"Constraint is not exactly pinned: {line}"
        parsed[_normalise(match.group("name"))] = match.group("version")
    return parsed


def test_dependency_constraints_match_project_metadata() -> None:
    assert _constraint_requirements() == _project_requirements()
