"""Copying model artifacts into a job's models root."""

from __future__ import annotations

import hashlib
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from ...core.inference.model_paths import copy_model_metadata_sidecars
from ..project_bundle import write_json_atomic
from .manifest import JobModel, TrackingJobError, validate_job_relpath

# Host-local engine caches never travel: they are rebuilt per compute box.
# Fix V-minor: keep this in sync with content_id.py's _EXCLUDED_DIR_NAMES --
# .DS_Store/__pycache__ are OS/tooling noise, not model content, and copying
# them into the job would be dead weight `directory_content_id` already
# ignores on the read side.
EXCLUDED_DIR_NAMES = {".hydra-runtime-artifacts", ".DS_Store", "__pycache__"}


@dataclass(frozen=True)
class PlannedModel:
    """One model to ship, already resolved by the app layer.

    ``key`` is the models-root-relative path the CONFIG already uses
    (``make_model_path_relative``), never re-derived from ``role``: the repo has
    two incompatible role->directory layouts over one registry, and preserving
    the config's own key is what makes the sidecar resolve on the remote.
    ``bundle_artifacts`` are sibling files found by ``discover_multihead_model_bundle``
    in the app layer (ClassKit is an app layer; Data must not import it).
    """

    role: str
    source_path: str
    kind: str  # "file" | "directory"
    key: str
    bundle_artifacts: list[str] = field(default_factory=list)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def external_key_for(path: str) -> str:
    """Job key for a model that lives OUTSIDE the staging models root."""
    source = Path(path)
    digest = hashlib.sha256(str(source.resolve()).encode("utf-8")).hexdigest()[:12]
    return f"external/{digest}/{source.name}"


def _copy_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def copy_model_reference(planned: PlannedModel, models_root: Path) -> JobModel:
    """Copy one model (plus sidecars/bundle siblings) into ``models_root``."""
    validate_job_relpath(planned.key)
    source = Path(planned.source_path)
    destination = Path(models_root) / planned.key
    if not source.exists():
        raise TrackingJobError(
            f"model for role {planned.role} does not exist on this machine: "
            f"{planned.source_path}",
            code=2,
        )

    sidecars: list[str] = []
    files: list[str] = []

    if planned.kind == "directory":
        if destination.exists():
            shutil.rmtree(destination)
        shutil.copytree(
            source,
            destination,
            ignore=shutil.ignore_patterns(*EXCLUDED_DIR_NAMES),
        )
        # Fix B4: file_digests was DECLARED on JobModel but never populated,
        # which made verify_job's "if model.file_digests: check every member"
        # branch vacuously true for every directory model that has ever been
        # packed. Populate it here -- this walk is the ONLY place that sees the
        # member list, so there is nowhere else it could be filled in.
        digests: dict[str, str] = {}
        for child in sorted(destination.rglob("*")):
            if child.is_file():
                relpath = child.relative_to(models_root).as_posix()
                files.append(relpath)
                digests[relpath] = _sha256(child)
        return JobModel(
            key=planned.key,
            roles=[planned.role],
            origin_path=str(source),
            kind="directory",
            # A directory has NO meaningful top-level sha256/size_bytes; both
            # stay at their dataclass defaults and verify_job must not check
            # them for kind == "directory" (see Task 7's verify rules).
            files=files,
            file_digests=digests,
        )

    _copy_file(source, destination)
    copy_model_metadata_sidecars(str(source), str(destination))
    for sidecar in sorted(destination.parent.glob(destination.name + ".*")):
        if sidecar != destination:
            sidecars.append(sidecar.relative_to(models_root).as_posix())
    v2 = destination.with_suffix(".v2meta.json")
    if v2.exists():
        sidecars.append(v2.relative_to(models_root).as_posix())

    for artifact in planned.bundle_artifacts:
        artifact_path = Path(artifact)
        if not artifact_path.exists():
            raise TrackingJobError(
                f"bundle artifact for role {planned.role} is missing: {artifact}",
                code=2,
            )
        sibling = destination.parent / artifact_path.name
        _copy_file(artifact_path, sibling)
        copy_model_metadata_sidecars(str(artifact_path), str(sibling))
        files.append(sibling.relative_to(models_root).as_posix())
        # Fix V1: bundle heads (ClassKit multi-head classifiers) carry their OWN
        # metadata sidecars, same two conventions as the primary model above
        # (three-suffix-appended and .v2meta.json). Without collecting these into
        # `sidecars`, a head's .v2meta.json is copied to disk but never recorded
        # in JobModel, so build_push_input_list (Task 9) never pushes it and
        # verify_job (Task 7) never checks it exists. A missing .v2meta.json for
        # a flat .pt classifier makes backend.py:288's with_suffix lookup miss,
        # falling into resolve_fit_policy(None, ...) -> "assuming legacy 'squash'"
        # (backend.py:38-52) -- a silently different crop preprocessing on the
        # remote, with only a warning logged.
        for head_sidecar in sorted(sibling.parent.glob(sibling.name + ".*")):
            if head_sidecar != sibling:
                sidecars.append(head_sidecar.relative_to(models_root).as_posix())
        head_v2 = sibling.with_suffix(".v2meta.json")
        if head_v2.exists():
            sidecars.append(head_v2.relative_to(models_root).as_posix())

    # Fix B4: the bundle artifacts are extra FILES beside the primary model;
    # the primary's own sha256 says nothing about them, so digest each one.
    # (Sidecars are metadata JSON regenerated by copy_model_metadata_sidecars
    # and are deliberately not digested -- verify only asserts they exist.)
    file_digests = {
        relpath: _sha256(Path(models_root) / relpath) for relpath in sorted(set(files))
    }

    return JobModel(
        key=planned.key,
        roles=[planned.role],
        origin_path=str(source),
        kind="file",
        sha256=_sha256(destination),
        size_bytes=destination.stat().st_size,
        sidecars=sorted(set(sidecars)),
        files=sorted(set(files)),
        file_digests=file_digests,
    )


def write_registry_subset(
    shipped_keys: set[str],
    entries: Iterable[tuple[str, dict]],
    destination: Path,
) -> int:
    """Write a v2 registry containing only the shipped models.

    ``source_path`` is nulled: it is the staging machine's absolute path to the
    training artifact and is meaningless on any other host.
    """
    subset: dict[str, dict] = {}
    for key, meta in entries:
        if key in shipped_keys:
            entry = dict(meta)
            entry["source_path"] = None
            subset[key] = entry
    write_json_atomic(Path(destination), {"schema_version": 2, "entries": subset})
    return len(subset)
