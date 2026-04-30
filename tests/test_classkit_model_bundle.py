def test_shared_trunk_role_publishes_single_artifact():
    """Service layer must not include the shared-trunk role in its multi-head
    manifest set: the .pth alone carries factor_names."""
    from hydra_suite.training import service as svc
    from hydra_suite.training.contracts import TrainingRole

    multihead_role_sets = [
        v
        for k, v in vars(svc).items()
        if isinstance(v, (set, frozenset))
        and any(
            getattr(item, "name", "").startswith("CLASSIFY_MULTIHEAD") for item in v
        )
    ]
    assert multihead_role_sets, "expected a multi-head role set in service.py"
    for role_set in multihead_role_sets:
        assert (
            TrainingRole.CLASSIFY_MULTIHEAD_CUSTOM_SHARED not in role_set
        ), f"shared-trunk role must publish via single-artifact path; found in {role_set}"


def test_color_tag_preset_lists_shared_trunk_mode():
    from hydra_suite.classkit.config.presets import color_tag_preset

    scheme = color_tag_preset(2, ["red", "blue", "green"])
    assert "multihead_custom_shared" in scheme.training_modes
    scheme1 = color_tag_preset(1, ["red", "blue"])
    # Single-factor schemes still only allow flat modes
    assert "multihead_custom_shared" not in scheme1.training_modes


def test_task_workers_maps_shared_trunk_mode_to_role():
    from hydra_suite.classkit.jobs import task_workers
    from hydra_suite.training.contracts import TrainingRole

    mapping = getattr(task_workers, "MODE_TO_ROLE", None)
    if mapping is None:
        get_role = getattr(task_workers, "training_role_for_mode", None)
        if get_role is None:
            # Mapping currently lives on the GUI MainWindow as a static method.
            # Verify the role exists in that mapping instead.
            from hydra_suite.classkit.gui.main_window import MainWindow as _MW

            assert (
                _MW._training_role_for_mode("multihead_custom_shared")
                == TrainingRole.CLASSIFY_MULTIHEAD_CUSTOM_SHARED
            )
            return
        assert (
            get_role("multihead_custom_shared")
            == TrainingRole.CLASSIFY_MULTIHEAD_CUSTOM_SHARED
        )
    else:
        assert (
            mapping["multihead_custom_shared"]
            == TrainingRole.CLASSIFY_MULTIHEAD_CUSTOM_SHARED
        )
