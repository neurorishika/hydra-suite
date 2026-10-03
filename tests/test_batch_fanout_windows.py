"""Windows code paths of the batch GPU fan-out (exercised on any OS via seams)."""

from __future__ import annotations

import signal
import types

import pytest

from hydra_suite.trackerkit import batch_fanout as bf


class _FakeProc:
    def __init__(self, pid, children=()):
        self.pid = pid
        self._children = list(children)
        self.killed = False

    def children(self, recursive=False):
        return list(self._children)

    def kill(self):
        self.killed = True


@pytest.fixture
def fake_psutil(monkeypatch):
    import psutil

    grandchild = _FakeProc(30)
    child = _FakeProc(20, [grandchild])
    table = {20: child}

    def process(pid):
        if pid not in table:
            raise psutil.NoSuchProcess(pid)
        return table[pid]

    monkeypatch.setattr(psutil, "Process", process)
    monkeypatch.setattr(psutil, "wait_procs", lambda procs, timeout=None: ([], []))
    monkeypatch.setattr(bf, "_is_windows", lambda: True)
    return types.SimpleNamespace(child=child, grandchild=grandchild)


def test_sigkill_has_a_windows_stand_in():
    assert bf._SIGKILL == getattr(signal, "SIGKILL", signal.SIGTERM)


def test_windows_signal_group_kills_the_whole_live_tree(fake_psutil):
    popen = types.SimpleNamespace(
        pid=20, kill=lambda: pytest.fail("tree kill suffices")
    )
    bf._signal_group(popen, bf._SIGKILL)
    assert fake_psutil.child.killed and fake_psutil.grandchild.killed


def test_windows_signal_group_falls_back_when_leader_is_gone(fake_psutil):
    hit = []
    popen = types.SimpleNamespace(pid=999, kill=lambda: hit.append("kill"))
    bf._signal_group(popen, bf._SIGKILL)
    assert hit == ["kill"]


def test_windows_kill_all_reports_and_prunes(fake_psutil):
    registry = bf.LiveChildRegistry()
    registry.add(20)
    registry.add(999)
    assert registry.kill_all() == [20]
    assert 999 not in registry.pids()


def test_stop_signals_include_sigbreak_when_the_platform_has_it(monkeypatch):
    from hydra_suite.trackerkit import cli

    monkeypatch.setattr(signal, "SIGBREAK", 21, raising=False)
    assert 21 in cli._fanout_stop_signals()
