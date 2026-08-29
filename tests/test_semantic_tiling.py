import numpy as np

from hydra_suite.core.inference.semantic.base import SemanticInstance, SemanticLabeler


class FakeLabeler:
    """Returns a scripted list of instances per call, in TILE-LOCAL coords."""

    def __init__(self, scripted: list[list[SemanticInstance]]) -> None:
        self._scripted = list(scripted)
        self.calls = 0

    @property
    def name(self) -> str:
        return "fake"

    def label_image(
        self, image_bgr, prompt, *, confidence_threshold=0.0, max_instances=0
    ):
        out = self._scripted[self.calls] if self.calls < len(self._scripted) else []
        self.calls += 1
        return [i for i in out if i.confidence >= confidence_threshold]


def test_fake_labeler_satisfies_the_protocol():
    assert isinstance(FakeLabeler([]), SemanticLabeler)


def test_semantic_instance_is_frozen():
    inst = SemanticInstance(
        polygon_px=np.zeros((4, 2), dtype=np.float32), confidence=0.5
    )
    try:
        inst.confidence = 0.9
    except Exception as exc:
        # dataclasses.FrozenInstanceError's message text is always
        # "cannot assign to field '...'" -- it never contains the word
        # "frozen" -- only the exception TYPE name does. Check the type.
        assert "frozen" in type(exc).__name__.lower()
    else:
        raise AssertionError("SemanticInstance must be frozen")
