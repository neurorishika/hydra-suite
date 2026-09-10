"""Shared builders for the portable-job tests."""


def _planned(staging, **overrides):
    from hydra_suite.data.tracking_job.pack import PlannedVideo
    from hydra_suite.data.tracking_job.references import PlannedModel

    config = {
        "file_path": str(staging["video"]),
        "csv_path": str(staging["video"].with_name("colony_tracking.csv")),
        "video_output_path": str(staging["video"].with_name("colony_tracking.mp4")),
        "yolo_obb_direct_model_path": "obb/x.pt",
        "pose_skeleton_file": str(staging["skeleton"]),
    }
    config.update(overrides.pop("config", {}))
    # Fix (against real Python semantics, not the plan's literal text): the
    # plan's own `_planned` hardcodes `planned_models=` below while also
    # forwarding `**overrides`, so any caller passing `planned_models=`
    # (several tests in this task do) raises "got multiple values for
    # keyword argument 'planned_models'". Pop it the same way `config` is
    # popped just above.
    planned_models = overrides.pop(
        "planned_models",
        [
            PlannedModel(
                role="YOLO_OBB_DIRECT_MODEL_PATH",
                source_path=str(staging["models"] / "obb" / "x.pt"),
                kind="file",
                key="obb/x.pt",
            )
        ],
    )
    video_path = overrides.pop("video_path", str(staging["video"]))
    return PlannedVideo(
        video_path=video_path,
        config=config,
        config_provenance="own-sidecar",
        planned_models=planned_models,
        skeleton_path=str(staging["skeleton"]),
        **overrides,
    )
