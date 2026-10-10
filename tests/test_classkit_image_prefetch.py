import os
import time

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
cv2 = pytest.importorskip("cv2")
pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication  # noqa: E402

from hydra_suite.classkit.gui.widgets.image_viewer import ImageCanvas  # noqa: E402


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def _write(tmp_path, n):
    paths = []
    for i in range(n):
        p = tmp_path / f"{i}.png"
        cv2.imwrite(str(p), np.full((16, 24, 3), i * 10, dtype=np.uint8))
        paths.append(str(p))
    return paths


def _wait(app, cond, timeout=5.0):
    end = time.time() + timeout
    while time.time() < end and not cond():
        app.processEvents()
        time.sleep(0.01)
    return cond()


def test_prefetch_fills_cache(app, tmp_path):
    canvas = ImageCanvas()
    paths = _write(tmp_path, 6)
    assert canvas.prefetch(paths) == 6
    assert _wait(app, lambda: all(p in canvas._pixmap_cache for p in paths))
    assert canvas.prefetch(paths) == 0  # all cached: nothing to queue
    canvas.set_image(paths[3])
    assert canvas._last_path == paths[3]


def test_prefetch_discards_results_after_enhancement_change(app, tmp_path):
    canvas = ImageCanvas()
    paths = _write(tmp_path, 3)
    canvas.prefetch(paths)
    canvas.use_clahe = True  # settings change while loads are in flight
    _wait(app, lambda: False, timeout=0.5)
    assert not any(p in canvas._pixmap_cache for p in paths)


def test_prefetch_clahe_loads(app, tmp_path):
    canvas = ImageCanvas()
    canvas.use_clahe = True
    paths = _write(tmp_path, 2)
    canvas.prefetch(paths)
    assert _wait(app, lambda: all(p in canvas._pixmap_cache for p in paths))


def test_missing_file_is_ignored(app, tmp_path):
    canvas = ImageCanvas()
    canvas.prefetch([str(tmp_path / "nope.png")])
    _wait(app, lambda: False, timeout=0.3)
    assert not canvas._pixmap_cache
