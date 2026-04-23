"""Preset LabelingScheme helpers for built-in and user-saved ClassKit schemes."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path

from hydra_suite.paths import get_classkit_scheme_presets_dir

from .schemas import Factor, LabelingScheme

logger = logging.getLogger(__name__)

DEFAULT_COLOR_TAG_COLORS = ["red", "blue", "green", "yellow", "white"]


@dataclass(frozen=True)
class SchemePreset:
    """Resolved ClassKit scheme preset metadata."""

    key: str
    label: str
    description: str
    scheme: LabelingScheme
    is_custom: bool = False
    path: Path | None = None


def head_tail_preset() -> LabelingScheme:
    """Single-factor, 4-class head/tail direction classifier."""
    return LabelingScheme(
        name="head_tail",
        factors=[
            Factor(
                name="direction",
                labels=["left", "right", "up", "down"],
                shortcut_keys=["a", "d", "w", "s"],
            )
        ],
        training_modes=["flat_yolo", "flat_custom"],
        description="Head/tail orientation: left, right, up, down",
    )


def color_tag_preset(n_factors: int, colors: list[str]) -> LabelingScheme:
    """Multi-factor ordered color-tag classifier.

    Args:
        n_factors: Number of ordered color tag positions (1, 2, or 3 typical).
        colors: Ordered list of color label names for each factor.
    """
    if n_factors < 1:
        raise ValueError("n_factors must be >= 1")
    if not colors:
        raise ValueError("colors must be non-empty")

    factors = [
        Factor(name=f"tag_{i + 1}", labels=list(colors)) for i in range(n_factors)
    ]
    if n_factors == 1:
        modes = ["flat_yolo", "flat_custom"]
    else:
        modes = [
            "flat_yolo",
            "flat_custom",
            "multihead_yolo",
            "multihead_custom",
        ]

    scheme = LabelingScheme(
        name=f"color_tags_{n_factors}factor",
        factors=factors,
        training_modes=modes,
        description="",
    )
    if n_factors == 1:
        scheme.description = f"1-factor color tag: {len(colors)} colors"
    else:
        scheme.description = (
            f"{n_factors}-factor color tag: {len(colors)} colors + unknown each "
            f"→ {scheme.total_classes} composites"
        )
    return scheme


def age_preset(extra_classes: list[str] | None = None) -> LabelingScheme:
    """Single-factor age classifier (young/old), extensible."""
    labels = ["young", "old"] + list(extra_classes or [])
    return LabelingScheme(
        name="age",
        factors=[Factor(name="age", labels=labels)],
        training_modes=["flat_yolo", "flat_custom"],
        description="Age classification: " + ", ".join(labels),
    )


def apriltag_preset(family: str, max_tag_id: int) -> LabelingScheme:
    """Single-factor scheme for AprilTag ID classification.

    Args:
        family: AprilTag family string (e.g. 'tag36h11').
        max_tag_id: Highest tag ID to include. Labels will be tag_0..tag_N + no_tag.
    """
    labels = [f"tag_{i}" for i in range(max_tag_id + 1)] + ["no_tag"]
    return LabelingScheme(
        name=f"apriltag_{family}",
        factors=[Factor(name=family, labels=labels)],
        training_modes=["flat_yolo", "flat_custom"],
        description=f"AprilTag {family} classifier: tag_0..tag_{max_tag_id} + no_tag",
    )


def flatten_scheme_labels(scheme: LabelingScheme | dict) -> list[str]:
    """Return a flat list of all labels defined across a scheme's factors."""
    scheme_obj = _coerce_scheme(scheme)
    labels: list[str] = []
    for factor in scheme_obj.factors:
        labels.extend(factor.labels)
    return labels


def describe_scheme(scheme: LabelingScheme | dict) -> str:
    """Build a compact human-readable summary for a scheme."""
    scheme_obj = _coerce_scheme(scheme)
    if not scheme_obj.factors:
        return "No factors defined."
    if len(scheme_obj.factors) == 1:
        labels = ", ".join(scheme_obj.factors[0].labels)
        return f"1 factor · {len(scheme_obj.factors[0].labels)} labels: {labels}."
    return (
        f"{len(scheme_obj.factors)} factors · "
        f"{scheme_obj.total_classes} composite classes."
    )


def get_builtin_scheme_presets() -> list[SchemePreset]:
    """Return the built-in ClassKit labeling presets."""
    color_tag_1 = color_tag_preset(1, DEFAULT_COLOR_TAG_COLORS)
    color_tag_2 = color_tag_preset(2, DEFAULT_COLOR_TAG_COLORS)
    color_tag_3 = color_tag_preset(3, DEFAULT_COLOR_TAG_COLORS)
    return [
        SchemePreset(
            key="head_tail",
            label="Head / Tail  (4 directions · A D W S)",
            description="1 factor · 4 labels: left, right, up, down  (keys A D W S).",
            scheme=head_tail_preset(),
        ),
        SchemePreset(
            key="color_tag_1",
            label="Color tag — 1 factor  (5 colors)",
            description=f"1 factor · 5 labels: {', '.join(DEFAULT_COLOR_TAG_COLORS)}.",
            scheme=color_tag_1,
        ),
        SchemePreset(
            key="color_tag_2",
            label=f"Color tag — 2 factors  ({color_tag_2.total_classes} composites)",
            description=(
                "2 factors × (5 colors + unknown) "
                f"= {color_tag_2.total_classes} composite labels."
            ),
            scheme=color_tag_2,
        ),
        SchemePreset(
            key="color_tag_3",
            label=f"Color tag — 3 factors  ({color_tag_3.total_classes} composites)",
            description=(
                "3 factors × (5 colors + unknown) "
                f"= {color_tag_3.total_classes} composite labels."
            ),
            scheme=color_tag_3,
        ),
        SchemePreset(
            key="age",
            label="Age  (young / old)",
            description="1 factor · 2 labels: young, old.",
            scheme=age_preset(),
        ),
    ]


def list_saved_scheme_presets() -> list[SchemePreset]:
    """Load user-saved ClassKit scheme presets from the config directory."""
    presets: list[SchemePreset] = []
    for preset_path in sorted(get_classkit_scheme_presets_dir().glob("*.json")):
        try:
            payload = json.loads(preset_path.read_text(encoding="utf-8"))
            scheme_dict = payload.get("scheme", payload)
            scheme = LabelingScheme.from_dict(scheme_dict)
            label = payload.get("preset_name") or scheme.name
            description = payload.get("description") or describe_scheme(scheme)
        except (OSError, TypeError, ValueError, KeyError, json.JSONDecodeError) as exc:
            logger.warning(
                "Skipping invalid ClassKit scheme preset %s: %s", preset_path, exc
            )
            continue
        presets.append(
            SchemePreset(
                key=f"custom:{preset_path.stem}",
                label=f"Custom preset — {label}",
                description=description,
                scheme=scheme,
                is_custom=True,
                path=preset_path,
            )
        )
    return presets


def get_available_scheme_presets() -> list[SchemePreset]:
    """Return built-in presets followed by any saved user presets."""
    return get_builtin_scheme_presets() + list_saved_scheme_presets()


def save_scheme_preset(
    preset_name: str,
    scheme: LabelingScheme | dict,
    *,
    overwrite: bool = False,
) -> Path:
    """Persist a scheme preset to the user config directory."""
    normalized_name = preset_name.strip()
    if not normalized_name:
        raise ValueError("Preset name cannot be empty")

    preset_path = (
        get_classkit_scheme_presets_dir()
        / f"{_slugify_preset_name(normalized_name)}.json"
    )
    if preset_path.exists() and not overwrite:
        raise FileExistsError(preset_path)

    scheme_obj = _coerce_scheme(scheme)
    payload = {
        "preset_name": normalized_name,
        "description": describe_scheme(scheme_obj),
        "scheme": scheme_obj.to_dict(),
    }
    preset_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return preset_path


def get_custom_scheme_preset_key(preset_name: str) -> str:
    """Return the combo-box key used for a saved preset name."""
    return f"custom:{_slugify_preset_name(preset_name)}"


def _coerce_scheme(scheme: LabelingScheme | dict) -> LabelingScheme:
    if isinstance(scheme, LabelingScheme):
        return scheme
    return LabelingScheme.from_dict(scheme)


def _slugify_preset_name(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", name.strip().lower()).strip("_")
    if not slug:
        raise ValueError("Preset name must include at least one letter or number")
    return slug
