"""Every span name, as a constant.

Bare string literals at call sites would be copy-pasted boilerplate in string
form, and a refactor that moved a function would silently drop its row with
nothing failing. ``tests/utils/test_profiling_registry.py`` enforces both
halves: every constant here is used, and no call site passes a literal.

Names are LOCAL to their parent — the tree supplies the prefix, so
``CROP_EXTRACT`` under ``cnn`` and under ``pose`` stay distinct without
callers hand-prefixing strings. Dynamic / label-keyed names are prohibited:
they would make memory O(labels).
"""

from __future__ import annotations

import functools

from .profiling import span

# -- session tree ---------------------------------------------------------
SESSION = "session"
# NOTE: the spec map lists track_forward / track_backward here. They are
# omitted deliberately: the session profiler arms inside run_post_tracking
# (session.py:528), which runs AFTER both tracking passes complete, so there is
# no scope in which those spans could be opened. The passes are already
# profiled separately by worker.py's own profiler.
POSTPROCESS = "postprocess"
BACKWARD_POSTPROCESS = "backward_postprocess"
POSE_QUALITY = "pose_quality"
TEMPORAL_POSE = "temporal_pose"
TRAJECTORY_POSTPROC = "trajectory_postproc"
INTERPOLATE_AND_SCALE = "interpolate_and_scale"
MERGE = "merge"
RICH_EXPORT = "rich_export"
BUILD_DATAFRAME = "build_dataframe"
RELINK = "relink"
WRITE = "write"
DATASET_GENERATION = "dataset_generation"
MEDIA_EXPORT = "media_export"
ANNOTATED_VIDEO = "annotated_video"

# -- inference tree -------------------------------------------------------
INFERENCE = "inference"
BATCH_PASS = "batch_pass"
OPEN_CACHES = "open_caches"
WINDOW = "window"
DECODE = "decode"
DETECT = "detect"
RUN_OBB = "run_obb"
MODEL_EXECUTE = "model_execute"
EXTRACT_RAW = "extract_raw"
MATERIALIZE = "materialize"
RUN_BGSUB_BATCH = "run_bgsub_batch"
FILTER = "filter"
HEADTAIL = "headtail"
CNN = "cnn"
POSE = "pose"
CROP_EXTRACT = "crop_extract"
FRAME_TO_CHW = "frame_to_chw"
AFFINE_LOOP = "affine_loop"
WARP_BATCH = "warp_batch"
FOREIGN_MASK = "foreign_mask"
APPLY_FIT = "apply_fit"
BACKEND_FORWARD = "backend_forward"
PREP_LOOP = "prep_loop"
APRILTAG = "apriltag"
CACHE_WRITE = "cache_write"
ENQUEUE = "enqueue"
FLUSH = "flush"
ASSEMBLE_SCATTER = "assemble_scatter"

# -- realtime tree --------------------------------------------------------
REALTIME = "realtime"
RT_OBB = "obb"
RT_CROPS = "crops"
RT_INDIVIDUAL = "individual"
RT_CACHE = "cache"
RT_FINALIZE = "finalize"

# -- post tree ------------------------------------------------------------
POST = "post"
PREPARE = "prepare"
RESOLVE = "resolve"
INTERPOLATE = "interpolate"
TAG_IDENTITY = "tag_identity"
RESCALE = "rescale"

# -- interpolated-crops tree ---------------------------------------------
INTERP_CROPS = "interp_crops"
SETUP = "setup"
GAP_DETECTION = "gap_detection"
CROP_EXTRACTION = "crop_extraction"
READ = "read"
WARP = "warp"
POSE_INFERENCE = "pose_inference"
CNN_INFERENCE = "cnn_inference"
FINALIZE = "finalize"


def spanned(name: str, units: float | None = None, gpu: bool = False):
    """Wrap a function body in a span. Use for function boundaries.

    ``with span(...)`` is reserved for sub-function regions such as
    ``FRAME_TO_CHW`` and ``AFFINE_LOOP``.
    """

    def _decorate(fn):
        @functools.wraps(fn)
        def _wrapper(*args, **kwargs):
            with span(name, units=units, gpu=gpu):
                return fn(*args, **kwargs)

        return _wrapper

    return _decorate
