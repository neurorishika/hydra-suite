"""Process-isolated memory regression for the SAM3 descriptor data path."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

_MEASURE_SCRIPT = textwrap.dedent("""
    import json
    import resource
    import sys

    import numpy as np

    import hydra_suite.training.sam3_lora.dataloader as dl

    count = int(sys.argv[1])
    descriptors = [
        dl.TileDescriptor(
            image_id=i,
            image_path=str(i),
            positive_prompt="ant",
            negative_prompts=("floor", "wall", "food"),
            instances=(),
        )
        for i in range(count)
    ]
    dl.cv2.imread = lambda _path: np.zeros((1, 1, 3), dtype=np.uint8)
    dl._default_transform = lambda: object()

    def build(*_args):
        # Commit the same 11.63 MiB of pages as one real 1008 RGB float32
        # transform. Negatives are metadata queries and create no copies.
        image = np.ones((3, 1008, 1008), dtype=np.float32)
        return [image, image, image, image]

    dl.build_shared_query_datapoints = build
    dl.collate_datapoints = lambda values: np.stack(values)
    batches = dl.collate_batches(descriptors, batch_size=2)
    checksum = 0.0
    while True:
        try:
            batch = next(batches)
        except StopIteration:
            break
        checksum += float(batch[0, 0, 0, 0])
        del batch

    # ru_maxrss is bytes on macOS and KiB on Linux.
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform != "darwin":
        peak *= 1024
    print(json.dumps({"peak": peak, "checksum": checksum}))
    """)


def _peak_rss(tile_count: int) -> int:
    env = os.environ.copy()
    source_root = str(Path(__file__).resolve().parents[1] / "src")
    existing_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        os.pathsep.join((source_root, existing_pythonpath))
        if existing_pythonpath
        else source_root
    )
    completed = subprocess.run(
        [sys.executable, "-c", _MEASURE_SCRIPT, str(tile_count)],
        check=True,
        capture_output=True,
        env=env,
        text=True,
    )
    result = json.loads(completed.stdout.strip().splitlines()[-1])
    assert result["checksum"] > 0
    return int(result["peak"])


def test_peak_image_rss_is_bounded_by_batch_not_dataset_size():
    small_peak = _peak_rss(4)
    large_peak = _peak_rss(80)

    # Eighty eager tensors would add about 930 MiB. Allow 96 MiB of allocator
    # and process noise while requiring dataset-size-independent residency.
    assert large_peak <= small_peak + 96 * 1024 * 1024
