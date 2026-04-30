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
