import numpy as np

from hydra_suite.core.identity.classification.cnn import ClassPrediction
from hydra_suite.core.tracking.features.live_features import (
    LiveCNNIdentityStore,
    LivePosePropertiesStore,
    LiveTagObservationStore,
)


def test_live_pose_properties_store_exposes_cache_like_frame_view() -> None:
    store = LivePosePropertiesStore()
    store.update_frame(
        12,
        detection_ids=[7],
        pose_keypoints=[
            np.array(
                [
                    [10.0, 20.0, 0.9],
                    [11.0, 21.0, 0.8],
                    [12.0, 22.0, 0.1],
                ],
                dtype=np.float32,
            )
        ],
    )

    frame = store.get_frame(12, min_valid_conf=0.2)

    assert frame["detection_ids"] == [7]
    assert len(frame["pose_keypoints"]) == 1
    assert frame["pose_num_valid"] == [2]
    assert frame["pose_num_keypoints"] == [3]
    assert frame["pose_valid_fraction"] == [2 / 3]


def test_live_tag_observation_store_exposes_cache_like_frame_view() -> None:
    store = LiveTagObservationStore()
    store.update_frame(
        3,
        tag_ids=[11],
        centers_xy=[(14.0, 18.0)],
        corners=[np.array([[1, 1], [2, 1], [2, 2], [1, 2]], dtype=np.float32)],
        det_indices=[0],
        hammings=[1],
    )

    frame = store.get_frame(3)

    assert frame["tag_ids"].tolist() == [11]
    assert frame["det_indices"].tolist() == [0]
    assert frame["hammings"].tolist() == [1]
    assert frame["centers_xy"].shape == (1, 2)
    assert frame["corners"].shape == (1, 4, 2)


def test_live_cnn_identity_store_exposes_cache_like_load() -> None:
    store = LiveCNNIdentityStore()
    predictions = [
        ClassPrediction(
            det_index=2,
            factor_names=("flat",),
            class_names=("alpha",),
            confidences=(0.95,),
        )
    ]
    store.update_frame(9, predictions)

    loaded = store.load(9)

    assert len(loaded) == 1
    assert loaded[0].class_name == "alpha"
    assert loaded[0].confidence == 0.95
    assert loaded[0].det_index == 2
    assert loaded[0] is not predictions[0]
