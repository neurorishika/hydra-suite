"""DetectKit's SAHI training/preview group on the shared widget (S4 Task 16).

``SliceSettingsGroup`` keeps its name, constructor and host contract
(``load_from``/``to_settings`` for YOLO, ``load_sam3_tiling``/
``to_sam3_tiling`` for SAM3) as a thin subclass of the shared
:class:`~hydra_suite.widgets.slice_settings.SliceSettingsWidget`
(``train_yolo`` / ``train_sam3`` roles). All translation goes through
``slice_settings_adapter``.

The two accessor pairs stay mode-gated by construction, so the two
independent ``full_frame_mix`` fields (YOLO dataset mix, SAM3 full-frame arm)
can never be cross-assigned. ``min_area_ratio`` stays live in both modes but
means something different below the floor (YOLO drops the instance; SAM3
marks it ``iscrowd`` -- decision D3, deliberately not unified); only its
wording is role-specific.
"""

from __future__ import annotations

from hydra_suite.utils.tiling_spec import DEFAULT_YOLO_IMGSZ
from hydra_suite.widgets.slice_settings import SliceSettingsWidget
from hydra_suite.widgets.tile_layout_preview import _TileLayoutPreview  # noqa: F401

from ..models import SliceTrainingSettings
from .slice_settings_adapter import (
    sam3_tiling_to_spec,
    settings_to_spec,
    spec_to_sam3_tiling,
    spec_to_settings,
)

# Old attribute name -> shared-widget attribute (decision 28). Read-only
# aliases so existing callers and tests keep working.
_LEGACY_ALIASES = {
    "chk_enabled": "chk_slice_enabled",
    "cmb_mode": "combo_slice_geometry",
    "txt_targets": "txt_slice_scales",
    "spin_w": "spin_slice_tile_w",
    "spin_h": "spin_slice_tile_h",
    "spin_overlap": "spin_slice_overlap",
    "spin_object_fraction": "spin_slice_object_fraction",
    "spin_min_area": "spin_slice_min_area",
    "spin_neg": "spin_slice_negative",
    "spin_merge": "spin_slice_merge",
    "chk_full": "chk_slice_full_frame_mix",
    "chk_balance_loss": "chk_slice_balance_loss",
    "spin_balance_power": "spin_slice_balance_power",
    "chk_keep_empty": "chk_slice_keep_empty",
}


def _alias(name: str) -> property:
    return property(lambda self: getattr(self, name), doc=f"Legacy alias of {name}.")


class SliceSettingsGroup(SliceSettingsWidget):
    """SAHI sliced-training settings for the YOLO (default) or SAM3 backend."""

    _DEFAULT_MODEL_INPUT_SIZE = DEFAULT_YOLO_IMGSZ
    _SAM3_MODEL_INPUT_SIZE = 1008

    def __init__(self, parent=None, *, backend: str = "yolo") -> None:
        if backend not in ("yolo", "sam3"):
            raise ValueError(f"unknown slice-settings backend: {backend!r}")
        self._backend = backend
        super().__init__(
            parent,
            role="train_sam3" if backend == "sam3" else "train_yolo",
            title=(
                "Tiling (SAHI geometry)"
                if backend == "sam3"
                else "Sliced dataset / inference (SAHI)"
            ),
        )
        self.set_model_input_size(
            self._SAM3_MODEL_INPUT_SIZE
            if backend == "sam3"
            else self._DEFAULT_MODEL_INPUT_SIZE
        )

    @property
    def backend(self) -> str:
        """Which training path this widget is wired to (``yolo`` or ``sam3``)."""
        return self._backend

    def _require_backend(self, backend: str, method: str) -> None:
        if self._backend != backend:
            raise RuntimeError(
                f"{method}() is only valid on a {backend!r} SliceSettingsGroup; "
                f"this one is {self._backend!r}."
            )

    # -- YOLO accessors ---------------------------------------------------

    def load_from(self, s: SliceTrainingSettings) -> None:
        self._require_backend("yolo", "load_from")
        spec, extras = settings_to_spec(s)
        self.set_spec(spec, extras=extras)

    def to_settings(self) -> SliceTrainingSettings:
        self._require_backend("yolo", "to_settings")
        return spec_to_settings(self.spec(), self.extras())

    # -- SAM3 accessors ---------------------------------------------------

    def load_sam3_tiling(
        self,
        *,
        geometry_mode: str,
        object_tile_fraction: float,
        object_tile_fractions,
        full_frame_mix: bool,
        slice_width: int,
        slice_height: int,
        tile_overlap: float,
        keep_empty_tiles: bool,
        min_area_ratio: float,
    ) -> None:
        """Load SAM3 tiling values (primitives, so the widget stays contract-free)."""
        self._require_backend("sam3", "load_sam3_tiling")
        spec, extras = sam3_tiling_to_spec(
            {
                "geometry_mode": geometry_mode,
                "object_tile_fraction": object_tile_fraction,
                "object_tile_fractions": object_tile_fractions,
                "full_frame_mix": full_frame_mix,
                "slice_width": slice_width,
                "slice_height": slice_height,
                "tile_overlap": tile_overlap,
                "keep_empty_tiles": keep_empty_tiles,
                "min_area_ratio": min_area_ratio,
            }
        )
        self.set_spec(spec, extras=extras)

    def to_sam3_tiling(self) -> dict:
        """Return the nine SAM3 tiling kwargs for ``Sam3LoraParams`` (no px list)."""
        self._require_backend("sam3", "to_sam3_tiling")
        return spec_to_sam3_tiling(self.spec(), self.extras())


for _old, _new in _LEGACY_ALIASES.items():
    setattr(SliceSettingsGroup, _old, _alias(_new))
del _old, _new
