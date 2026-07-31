"""InferenceRunner.obb_class_names exposes the loaded OBB model's id->name map
for label display, normalizing ultralytics' dict-or-list ``.names`` shape."""

from hydra_suite.core.inference.runner import InferenceRunner


def _make_runner(models_obj):
    runner = InferenceRunner.__new__(InferenceRunner)
    runner._models = models_obj
    return runner


class _Model:
    def __init__(self, names):
        self.names = names


class _OBB:
    def __init__(self, direct_model=None, obb_model=None):
        self.direct_model = direct_model
        self.obb_model = obb_model


class _Models:
    def __init__(self, obb=None):
        self.obb = obb


def test_obb_class_names_none_when_no_obb_model():
    runner = _make_runner(_Models(obb=None))
    assert runner.obb_class_names is None


def test_obb_class_names_normalizes_dict_names():
    obb = _OBB(direct_model=_Model({0: "ant", 1: "queen"}))
    runner = _make_runner(_Models(obb=obb))
    assert runner.obb_class_names == {0: "ant", 1: "queen"}


def test_obb_class_names_normalizes_list_names():
    obb = _OBB(direct_model=_Model(["ant", "queen"]))
    runner = _make_runner(_Models(obb=obb))
    assert runner.obb_class_names == {0: "ant", 1: "queen"}


def test_obb_class_names_falls_back_to_sequential_obb_model():
    obb = _OBB(direct_model=None, obb_model=_Model({0: "worker"}))
    runner = _make_runner(_Models(obb=obb))
    assert runner.obb_class_names == {0: "worker"}


def test_obb_class_names_none_when_model_has_no_names_attr():
    obb = _OBB(direct_model=object())
    runner = _make_runner(_Models(obb=obb))
    assert runner.obb_class_names is None
