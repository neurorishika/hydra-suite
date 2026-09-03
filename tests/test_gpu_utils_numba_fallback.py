"""The numba-absent import path.

The SAM3 training sidecar runs in a deliberately minimal conda env that has
no numba. `gpu_utils` used to set `njit = None` there, so `kalman.py`'s
module-scope `@njit(cache=True)` raised `TypeError: 'NoneType' object is not
callable` and made the whole `hydra_suite` package unimportable -- which
killed the sidecar before training started.
"""

import subprocess
import sys
import textwrap


def _run_without_numba(body: str) -> subprocess.CompletedProcess:
    """Run *body* in a subprocess where importing numba always fails."""
    script = textwrap.dedent("""
        import sys

        class _Blocker:
            def find_module(self, name, path=None):
                return None

            def find_spec(self, name, target=None, path=None):
                if name == "numba" or name.startswith("numba."):
                    raise ImportError("numba blocked for this test")
                return None

        sys.meta_path.insert(0, _Blocker())
        """) + textwrap.dedent(body)
    return subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True
    )


def test_njit_fallback_is_a_passthrough_not_none():
    result = _run_without_numba("""
        from hydra_suite.utils.gpu_utils import NUMBA_AVAILABLE, njit, prange

        assert NUMBA_AVAILABLE is False
        assert njit is not None

        # Both decorator spellings must work.
        @njit(cache=True)
        def with_args(x):
            return x * 2

        @njit
        def bare(x):
            return x + 1

        assert with_args(3) == 6
        assert bare(3) == 4
        assert list(prange(3)) == [0, 1, 2]
        print("OK")
        """)
    assert "OK" in result.stdout, result.stderr


def test_kalman_imports_without_numba():
    """The module that actually broke: unguarded @njit at module scope."""
    result = _run_without_numba("""
        import hydra_suite.core.filters.kalman as kalman

        assert kalman.NUMBA_AVAILABLE is False
        print("OK")
        """)
    assert "OK" in result.stdout, result.stderr


def test_sam3_sidecar_cli_imports_without_core_dependencies():
    """The sidecar env has no numba/sklearn/cv2/ultralytics.

    `python -m hydra_suite.training.sam3_lora.cli` imports the parent
    packages first, so an eager `.service` import in `training/__init__.py`
    dragged in all of `hydra_suite.core` and aborted every sidecar run.
    """
    blocked = ("numba", "sklearn", "cv2", "ultralytics")
    script = """
        import sys

        BLOCKED = %r

        class _Blocker:
            def find_spec(self, name, target=None, path=None):
                root = name.split(".")[0]
                if root in BLOCKED:
                    raise ImportError(f"{root} blocked for this test")
                return None

        sys.meta_path.insert(0, _Blocker())

        import hydra_suite.training as training

        assert training.TrainingRole.SEMANTIC_SAM3.value == "semantic_sam3"
        assert "service" not in sys.modules.get(
            "hydra_suite.training", training
        ).__dict__, "service was imported eagerly"
        print("OK")
    """ % (blocked,)
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script)],
        capture_output=True,
        text=True,
    )
    assert "OK" in result.stdout, result.stderr
