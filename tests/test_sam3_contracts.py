from hydra_suite.training.contracts import (
    Sam3LoraParams,
    SourceDataset,
    TrainingHyperParams,
    TrainingRole,
    TrainingRunSpec,
)


def test_role_exists_with_the_expected_value():
    assert TrainingRole.SEMANTIC_SAM3.value == "semantic_sam3"


def test_defaults_match_the_measured_spike():
    p = Sam3LoraParams(prompt="ant")
    assert p.rank == 16 and p.alpha == 32
    # 10, not 40: AP plateaus by epoch ~9 (sd 0.040 thereafter).
    assert p.epochs == 10
    # batch 1: batch 2 OOMed at 1008px on a 47 GB card.
    assert p.batch == 1 and p.grad_accum == 8
    # Adapting the text encoder risks eroding prompt discrimination.
    assert p.adapt_text_encoder is False
    assert p.negative_prompts == []
    assert p.label_quality_acknowledged is False


def test_spec_round_trips_sam3_params():
    spec = TrainingRunSpec(
        role=TrainingRole.SEMANTIC_SAM3,
        source_datasets=[SourceDataset(path="/tmp/x", level="polygon")],
        derived_dataset_dir="/tmp/d",
        base_model="sam3",
        hyperparams=TrainingHyperParams(),
        sam3_params=Sam3LoraParams(prompt="ant with color patch"),
    )
    d = spec.to_dict()
    assert d["role"] == "semantic_sam3"
    assert d["sam3_params"]["prompt"] == "ant with color patch"
