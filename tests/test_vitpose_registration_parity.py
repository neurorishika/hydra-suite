"""Regression test locking pose-parity model registration.

Settled decision: ViTPose registration stays at parity with YOLO-pose/SLEAP
(copy into models/pose/<Backend>/, filename metadata, picker listing) — no
``model_registry.json`` entry is written for ANY pose backend. This test
guards both halves of that parity so a future edit can't accidentally:

1. diverge ViTPose's on-disk directory layout from YOLO/SLEAP, or
2. reintroduce a ``register_yolo_model(...)`` call into the pose import path
   (registry unification is a deliberately deferred, separate spec).
"""

import inspect

from hydra_suite.core.inference import model_paths as model_utils
from hydra_suite.trackerkit.gui.orchestrators import config as cfgmod


def test_vitpose_pose_dir_parallels_yolo_and_sleap():
    for backend, leaf in [("yolo", "YOLO"), ("sleap", "SLEAP"), ("vitpose", "ViTPose")]:
        assert str(model_utils.get_pose_models_directory(backend)).endswith(leaf)


def test_pose_import_does_not_call_yolo_registry():
    # Pose parity decision: NO pose backend writes model_registry.json.
    # `_import_pose_model_to_repository` is a method on ConfigOrchestrator
    # (not a module-level function), so we pull the source off the class.
    src = inspect.getsource(cfgmod.ConfigOrchestrator._import_pose_model_to_repository)
    assert "register_yolo_model" not in src, (
        "pose imports must stay registry-free (parity across yolo/sleap/vitpose); "
        "registry unification is a separate deferred spec"
    )
