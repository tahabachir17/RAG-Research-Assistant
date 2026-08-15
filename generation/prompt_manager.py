"""Load and render strict YAML/Jinja prompt templates."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml
from jinja2 import Environment, StrictUndefined, TemplateError, meta

try:
    from .entities import is_multi_paper_question
except ImportError:
    from entities import is_multi_paper_question


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
        if prompt.name == "qa_prompt":
            variables.setdefault("question_type_instruction", "")
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


_QUESTION_STEM = re.compile(
    r"\b(?:what|which|who|when|where|why|how|compare|contrast|describe|explain|"
    r"identify|list|report|evaluate|summarize)\b",
    re.IGNORECASE,
)


def detect_question_parts(question: str) -> list[str]:
    """Detect explicit multi-part questions with a conservative heuristic."""

    if not isinstance(question, str):
        raise TypeError("question must be a string")
    normalized = " ".join(question.split())
    if not normalized:
        raise ValueError("question must not be empty")
    marked = [part.strip(" ,;?") for part in re.findall(r"[^?]+\?", normalized)]
    marked = [part for part in marked if part]
    if len(marked) > 1:
        return marked
    if is_multi_paper_question(normalized):
        return [normalized.strip("?")]
    for match in re.finditer(r"\s+(?:and|as well as)\s+", normalized, re.IGNORECASE):
        left = normalized[: match.start()].strip(" ,;?")
        right = normalized[match.end() :].strip(" ,;?")
        if left and right and _QUESTION_STEM.search(left) and _QUESTION_STEM.search(right):
            return [left, right]
    return [normalized.strip("?")]


def compound_question_instruction(question: str) -> str:
    """Return an explicit coverage contract only for detected compound questions."""

    parts = detect_question_parts(question)
    if len(parts) < 2:
        return ""
    enumerated = " ".join(f"Part {index}: {part}" for index, part in enumerate(parts, 1))
    return (
        "This is a multi-part question. Address each part explicitly and separately, "
        "in this order, without fusing their answers: " + enumerated
    )


QUESTION_TYPE_INSTRUCTIONS = {
    "direct_fact": (
        "Give a short, direct answer before any supporting detail. Keep it short, "
        "but include any qualifying detail the context directly attaches to the "
        "requested fact, such as a condition, scope, or comparison."
    ),
    "mechanism": (
        "Enumerate each supported mechanism and explain what each one does. "
        "When the context names a specific mechanism with a technical term "
        "(e.g. a named layer, operation, or method), use that term in the answer "
        "rather than a generic paraphrase. If the context describes multiple "
        "distinct mechanisms, name each one individually rather than summarizing "
        "them as a category (e.g. 'parallelism and context'). For a multi-step "
        "process, trace the context-supported process from its inputs through "
        "intermediate results to its outcome. State what is stored between steps "
        "and what objective is optimized when the source specifies either; these "
        "are mechanisms, not optional supporting details."
    ),
    "causes_evidence": (
        "Enumerate each supported cause, then pair it with the reported evidence."
    ),
    "limitations_future_work": (
        "Separate current limitations from proposed future improvements."
    ),
    "comparison": (
        "Compare every paper explicitly named in the question along the same "
        "evidence-supported dimensions. Include every named paper. If its evidence "
        "is absent, say so explicitly instead of omitting it or guessing."
    ),
}


_ATTRIBUTION_PREFIX = re.compile(
    r"^[^?\n]*?(?:'[^']+'|\"[^\"]+\"|‘[^’]+’|“[^”]+”)\s*,\s*",
    re.IGNORECASE,
)


def _strip_attribution_prefix(question: str) -> str:
    """Remove a leading non-question clause that attributes a quoted paper."""

    stripped = question.strip()
    if _QUESTION_STEM.match(stripped):
        return stripped
    return _ATTRIBUTION_PREFIX.sub("", stripped, count=1)


def classify_question_type(question: str) -> str:
    """Classify a question into one of the five supported answer structures."""

    if not isinstance(question, str):
        raise TypeError("question must be a string")
    normalized = " ".join(_strip_attribution_prefix(question).casefold().split())
    if not normalized:
        raise ValueError("question must not be empty")
    if is_multi_paper_question(question) or re.search(
        r"\bcompare\b|\bcontrast\b|\bversus\b|\bvs\.?\b|\bdiffer", normalized
    ):
        return "comparison"
    if re.search(r"\blimitations?\b|\bfuture work\b|\bimprovements?\b", normalized):
        return "limitations_future_work"
    if normalized.startswith("why ") or re.search(
        r"\b(?:better|advantage|outperform|improv(?:e|es|ed|ement))\b", normalized
    ):
        return "causes_evidence"
    if re.match(r"how\s+(?:much|many|long|fast|often)\b", normalized):
        return "direct_fact"
    if normalized.startswith("how ") or re.search(
        r"\b(?:mechanisms?|work|operate|achieve)\b", normalized
    ):
        return "mechanism"
    return "direct_fact"


def question_type_instruction(question: str) -> str:
    """Return the distinct prompt fragment for a classified question."""

    question_type = classify_question_type(question)
    return f"Question type: {question_type}. {QUESTION_TYPE_INSTRUCTIONS[question_type]}"


def _normalize_name(template_name: str) -> str:
    if not isinstance(template_name, str):
        raise TypeError("template_name must be a string")
    name = template_name.strip()
    if name.endswith(".yaml"):
        name = name[:-5]
    if not re.fullmatch(r"[A-Za-z0-9_-]+", name):
        raise ValueError("template_name may contain only letters, numbers, '_' and '-'")
    return name
