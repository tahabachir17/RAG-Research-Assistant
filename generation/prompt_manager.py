"""Load and render strict YAML/Jinja prompt templates."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml
from jinja2 import Environment, StrictUndefined, TemplateError, meta


@dataclass(slots=True)
class PromptTemplate:
    name: str
    system: str
    user: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


class PromptManager:
    """Load validated prompt files once and render with strict variables."""

    def __init__(self, prompts_dir: str | Path = "config/prompts") -> None:
        self.prompts_dir = Path(prompts_dir)
        self._cache: dict[str, PromptTemplate] = {}
        self._environment = Environment(
            undefined=StrictUndefined, autoescape=False, keep_trailing_newline=True
        )

    def load(self, template_name: str) -> PromptTemplate:
        name = _normalize_name(template_name)
        if name in self._cache:
            return self._cache[name]
        path = self.prompts_dir / f"{name}.yaml"
        if not path.is_file():
            raise FileNotFoundError(f"Prompt template not found: {path}")
        try:
            payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        except yaml.YAMLError as exc:
            raise ValueError(f"Invalid YAML prompt template {path}: {exc}") from exc
        if not isinstance(payload, dict):
            raise ValueError(f"Prompt template {path} must be a YAML mapping")
        system, user = payload.get("system"), payload.get("user")
        if not isinstance(system, str) or not system.strip():
            raise ValueError(f"Prompt template {path} has no non-empty system string")
        if not isinstance(user, str) or not user.strip():
            raise ValueError(f"Prompt template {path} has no non-empty user string")
        template = PromptTemplate(name=name, system=system, user=user)
        self._cache[name] = template
        return template

    def render(self, template_name: str, **variables: Any) -> tuple[str, str]:
        prompt = self.load(template_name)
        required = set()
        try:
            for source in (prompt.system, prompt.user):
                required.update(
                    meta.find_undeclared_variables(self._environment.parse(source))
                )
        except TemplateError as exc:
            raise ValueError(
                f"Invalid Jinja syntax in prompt {prompt.name!r}: {exc}"
            ) from exc
        missing = sorted(name for name in required if name not in variables)
        if missing:
            raise ValueError(
                f"Missing variables for prompt {prompt.name!r}: {', '.join(missing)}"
            )
        try:
            system = self._environment.from_string(prompt.system).render(**variables)
            user = self._environment.from_string(prompt.user).render(**variables)
        except TemplateError as exc:
            raise ValueError(f"Could not render prompt {prompt.name!r}: {exc}") from exc
        if not system.strip() or not user.strip():
            raise ValueError(f"Rendered prompt {prompt.name!r} must not be blank")
        return system, user


def _normalize_name(template_name: str) -> str:
    if not isinstance(template_name, str):
        raise TypeError("template_name must be a string")
    name = template_name.strip()
    if name.endswith(".yaml"):
        name = name[:-5]
    if not re.fullmatch(r"[A-Za-z0-9_-]+", name):
        raise ValueError("template_name may contain only letters, numbers, '_' and '-'")
    return name
