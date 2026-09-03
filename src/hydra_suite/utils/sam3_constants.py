"""SAM3 geometry constants shared by inference, training, and publish.

Deliberately a dependency-free leaf module: it imports nothing. The SAM3
training sidecar runs in a minimal conda env with no numba, sklearn, cv2 or
ultralytics, and importing ``hydra_suite.core.inference.semantic.sam3`` for a
single integer executed ``hydra_suite/core/__init__.py``, which pulls in the
whole tracking stack (and, through ``data.al``, filterkit). One constant is
not worth that import graph -- and there must still be exactly ONE
definition, or train and serve can silently disagree on scale.
"""

# ultralytics' default cfg imgsz is 640 -- rounded up to 644 for the
# stride-14 backbone -- but build_sam3.py builds the SAM3 architecture at
# img_size=1008 and BasePredictor calls model.set_imgsz(self.imgsz).
# Inheriting the default therefore runs a 1008-native model at 644 with no
# warning. It also makes train/serve scale disagree for any finetuned
# checkpoint, which is what the sidecar's imgsz guard exists to catch.
PREDICTOR_IMGSZ = 1008
