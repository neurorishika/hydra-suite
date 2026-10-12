"""Zoomable image viewer widget with optional CLAHE enhancement."""

import cv2
import numpy as np
from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QBrush, QColor, QImage, QPainter, QPixmap, QWheelEvent
from PySide6.QtWidgets import QGraphicsPixmapItem, QGraphicsScene, QGraphicsView

from .image_prefetch import ImagePrefetcher


class ImageCanvas(QGraphicsView):
    """
    Styled image viewer based on PoseKit's PoseCanvas.
    Supports zooming, panning, and displaying an image.
    """

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setRenderHint(QPainter.Antialiasing, True)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)

        # PoseKit dark theme
        self.setBackgroundBrush(QBrush(QColor(18, 18, 18)))
        self.setStyleSheet("QGraphicsView { background-color: #121212; border: none; }")

        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)

        self.pix_item = QGraphicsPixmapItem()
        self.scene.addItem(self.pix_item)

        self._zoom_factor = 1.0
        self._min_zoom = 0.1
        self._max_zoom = 20.0
        self._pixmap_cache = {}
        self._cache_order = []
        self._cache_limit = 96
        self._last_path = None
        self._last_dimensions = None

        # Enhancement settings
        self._use_clahe = False
        self._clahe_clip = 2.0
        self._clahe_grid = (8, 8)

        self._prefetch_key = None
        self._prefetcher = ImagePrefetcher(self)
        self._prefetcher.image_ready.connect(self._on_prefetched)

    def prefetch(self, paths) -> int:
        """Warm the cache for ``paths`` on background threads (nearest first)."""
        paths = [str(p) for p in paths]
        if not paths:
            return 0
        use_clahe, clip, grid = self._enhancement_key
        self._prefetch_key = self._enhancement_key

        def loader(path: str):
            if not use_clahe:
                return QImage(path)
            img_bgr = cv2.imread(path)
            if img_bgr is None:
                return QImage(path)
            lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
            L, A, B = cv2.split(lab)
            L2 = cv2.createCLAHE(clipLimit=clip, tileGridSize=grid).apply(L)
            rgb = np.ascontiguousarray(
                cv2.cvtColor(cv2.merge([L2, A, B]), cv2.COLOR_LAB2RGB)
            )
            h, w, ch = rgb.shape
            return QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888).copy()

        return self._prefetcher.request(
            paths, loader, skip=lambda p: p in self._pixmap_cache
        )

    def _on_prefetched(self, path: str, image) -> None:
        if path in self._pixmap_cache or self._enhancement_key != self._prefetch_key:
            return  # already cached, or loaded under old enhancement settings
        self._cache_put(path, QPixmap.fromImage(image))

    @property
    def _enhancement_key(self):
        return (self._use_clahe, self._clahe_clip, self._clahe_grid)

    @property
    def use_clahe(self) -> bool:
        return self._use_clahe

    @use_clahe.setter
    def use_clahe(self, value: bool):
        if self._use_clahe != value:
            self._use_clahe = value
            # Clear cache when enhancement mode changes to force reload
            self.clear_cache()

    def set_clahe_params(self, clip: float, grid: tuple[int, int]):
        """Update CLAHE parameters and clear cache."""
        self._clahe_clip = float(clip)
        self._clahe_grid = (int(grid[0]), int(grid[1]))
        self.clear_cache()

    def clear_cache(self):
        """Clear the internal pixmap cache."""
        self._pixmap_cache.clear()
        self._cache_order.clear()

        # Force the next refresh to reload even if the same image/key is requested.
        self._last_path = None

    def _cache_get(self, key: str):
        if key not in self._pixmap_cache:
            return None
        return self._pixmap_cache[key]

    def _cache_put(self, key: str, pixmap: QPixmap):
        if key in self._pixmap_cache:
            return
        self._pixmap_cache[key] = pixmap
        self._cache_order.append(key)
        if len(self._cache_order) > self._cache_limit:
            old_key = self._cache_order.pop(0)
            self._pixmap_cache.pop(old_key, None)

    def set_image(self, img_path: str):
        """Load an image from path and display it."""
        if not img_path:
            return

        if (
            self._last_path == img_path
            and self.pix_item.pixmap()
            and not self.pix_item.pixmap().isNull()
        ):
            return

        cached = self._cache_get(img_path)
        if cached is not None:
            self._display_pixmap(cached)
            self._last_path = img_path
            return

        if self._use_clahe:
            # Load with OpenCV for processing
            img_bgr = cv2.imread(str(img_path))
            if img_bgr is not None:
                pixmap = self._cv_to_pixmap(img_bgr)
            else:
                pixmap = QPixmap(str(img_path))
        else:
            pixmap = QPixmap(str(img_path))

        if pixmap.isNull():
            return

        self._cache_put(img_path, pixmap)
        self._display_pixmap(pixmap)
        self._last_path = img_path

    def set_cv_image(self, img_rgb: np.ndarray, cache_key: str = None):
        """Display an RGB numpy array, applying enhancement if enabled."""
        if img_rgb is None:
            return

        if cache_key:
            cached = self._cache_get(cache_key)
            if cached is not None:
                self._display_pixmap(cached)
                self._last_path = cache_key
                return

        if self._use_clahe:
            # Apply CLAHE on L channel (LAB)
            # Input is RGB, convert to LAB
            lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
            L, A, B = cv2.split(lab)
            clahe = cv2.createCLAHE(
                clipLimit=self._clahe_clip, tileGridSize=self._clahe_grid
            )
            L2 = clahe.apply(L)
            lab2 = cv2.merge([L2, A, B])
            processed_rgb = cv2.cvtColor(lab2, cv2.COLOR_LAB2RGB)
        else:
            processed_rgb = img_rgb

        h, w, ch = processed_rgb.shape
        bytes_per_line = ch * w
        qimg = QImage(processed_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)

        if cache_key:
            self._cache_put(cache_key, pixmap)
            self._last_path = cache_key

        self._display_pixmap(pixmap)

    def _cv_to_pixmap(self, img_bgr: np.ndarray) -> QPixmap:
        """Convert BGR image to enhanced RGB Pixmap."""
        # Apply CLAHE on L channel (LAB)
        lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
        L, A, B = cv2.split(lab)
        clahe = cv2.createCLAHE(
            clipLimit=self._clahe_clip, tileGridSize=self._clahe_grid
        )
        L2 = clahe.apply(L)
        lab2 = cv2.merge([L2, A, B])
        img_enhanced = cv2.cvtColor(lab2, cv2.COLOR_LAB2RGB)

        h, w, ch = img_enhanced.shape
        bytes_per_line = ch * w
        qimg = QImage(img_enhanced.data, w, h, bytes_per_line, QImage.Format_RGB888)
        return QPixmap.fromImage(qimg)

    def _display_pixmap(self, pixmap: QPixmap):
        """Internal helper to set pixmap and update scene rect."""
        self.pix_item.setPixmap(pixmap)
        self.scene.setSceneRect(QRectF(pixmap.rect()))
        dims = (pixmap.width(), pixmap.height())
        if self._last_dimensions != dims:
            self.fitInView(self.scene.sceneRect(), Qt.KeepAspectRatio)
            self._last_dimensions = dims

    def set_pixmap(self, pixmap: QPixmap):
        """Display a pre-built QPixmap directly (bypasses the path cache)."""
        if pixmap is None or pixmap.isNull():
            return
        self._display_pixmap(pixmap)
        self._last_path = None  # mark as not from a path so next set_image reloads

    def clear_image(self) -> None:
        """Clear the current preview image and reset view-tracking state."""
        self.pix_item.setPixmap(QPixmap())
        self.scene.setSceneRect(QRectF())
        self._last_path = None
        self._last_dimensions = None

    def wheelEvent(self, event: QWheelEvent) -> None:
        """Zoom logic from PoseKit."""
        delta = event.angleDelta().y()
        if delta == 0:
            return

        zoom_in = delta > 0
        factor = 1.15 if zoom_in else 1 / 1.15

        current_scale = self.transform().m11()
        new_scale = current_scale * factor

        if self._min_zoom <= new_scale <= self._max_zoom:
            self.scale(factor, factor)
