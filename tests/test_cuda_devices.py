from __future__ import annotations

import pytest

from hydra_suite.runtime.cuda_devices import (
    list_cuda_devices,
    parse_gpu_selectors,
    resolve_gpu_selectors,
)

_SMI = (
    "0, GPU-aaaa1111-0000-0000-0000-000000000000, NVIDIA RTX 6000 Ada Generation, Disabled\n"
    "1, GPU-bbbb2222-0000-0000-0000-000000000000, NVIDIA RTX 6000 Ada Generation, Disabled\n"
    "2, GPU-cccc3333-0000-0000-0000-000000000000, NVIDIA A100, Enabled\n"
)


def _fake_runner(_cmd):
    return _SMI


def test_list_devices_parses_and_skips_mig():
    devices = list_cuda_devices(runner=_fake_runner)
    assert [d.index for d in devices] == [0, 1]
    assert devices[0].uuid.startswith("GPU-aaaa1111")
    assert devices[1].name == "NVIDIA RTX 6000 Ada Generation"


def test_list_devices_empty_when_smi_missing():
    assert list_cuda_devices(runner=lambda _cmd: None) == []


def test_parse_selectors_expands_ranges_and_keeps_uuids():
    assert parse_gpu_selectors("0,2-4,GPU-abcd") == ["0", "2", "3", "4", "GPU-abcd"]
    assert parse_gpu_selectors(" auto ") == ["auto"]
    assert parse_gpu_selectors("") == []


def test_resolve_ordinals_and_uuid_prefix():
    devices = list_cuda_devices(runner=_fake_runner)
    picked = resolve_gpu_selectors(["1", "GPU-aaaa"], devices=devices)
    assert [d.index for d in picked] == [1, 0]


def test_resolve_auto_returns_all():
    devices = list_cuda_devices(runner=_fake_runner)
    assert resolve_gpu_selectors(["auto"], devices=devices) == devices


@pytest.mark.parametrize("bad", [["7"], ["GPU-zzzz"], ["0", "GPU-aaaa"], ["GPU-"]])
def test_resolve_rejects_unknown_ambiguous_and_duplicates(bad):
    devices = list_cuda_devices(runner=_fake_runner)
    with pytest.raises(ValueError):
        resolve_gpu_selectors(bad, devices=devices)


def test_resolve_with_no_devices_is_an_error():
    with pytest.raises(ValueError):
        resolve_gpu_selectors(["0"], devices=[])
