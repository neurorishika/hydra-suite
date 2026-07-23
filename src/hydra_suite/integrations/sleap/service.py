#!/usr/bin/env python3
"""
Pose inference utilities (subprocess + caching).
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import textwrap
import threading
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from hydra_suite.utils.conda_utils import popen_conda, run_conda


class PoseInferenceService:
    """PoseInferenceService API surface documentation."""

    CACHE_VERSION = 3

    def __init__(
        self,
        out_root: Path,
        keypoint_names: List[str],
        skeleton_edges: Optional[List[Tuple[int, int]]] = None,
    ):
        self.out_root = Path(out_root)
        self.keypoint_names = list(keypoint_names)
        self.skeleton_edges = list(skeleton_edges or [])
        kpt_sig_src = "|".join(self.keypoint_names) + f":{len(self.keypoint_names)}"
        self.kpt_sig = hashlib.sha1(kpt_sig_src.encode("utf-8")).hexdigest()[:12]
        self._cache_mem: Dict[str, Dict[str, List[Tuple[float, float, float]]]] = {}
        self._last_sleap_service_metrics: Dict[str, float] = {}

    def _weights_sig(self, weights_path: Path) -> Optional[str]:
        if not weights_path.exists() or not weights_path.is_file():
            return None
        stat = weights_path.stat()
        token = f"{weights_path.resolve()}|{stat.st_mtime_ns}|{stat.st_size}"
        return hashlib.sha1(token.encode("utf-8")).hexdigest()[:12]

    def _sleap_model_sig(self, model_dir: Path) -> Optional[str]:
        if not model_dir.exists() or not model_dir.is_dir():
            return None
        parts = []
        for name in ("best.ckpt", "training_config.yaml"):
            p = model_dir / name
            if p.exists() and p.is_file():
                stat = p.stat()
                parts.append(f"{p.resolve()}|{stat.st_mtime_ns}|{stat.st_size}")
        if parts:
            token = "|".join(parts)
            return hashlib.sha1(token.encode("utf-8")).hexdigest()[:12]
        try:
            stat = model_dir.stat()
            token = f"{model_dir.resolve()}|{stat.st_mtime_ns}"
        except Exception:
            token = str(model_dir.resolve())
        return hashlib.sha1(token.encode("utf-8")).hexdigest()[:12]

    def _model_sig(self, model_path: Path, backend: str) -> Optional[str]:
        backend = (backend or "yolo").lower()
        if backend == "sleap":
            return self._sleap_model_sig(model_path)
        return self._weights_sig(model_path)

    def _cache_key(self, model_path: Path, backend: str) -> str:
        m = self._model_sig(model_path, backend) or "missing"
        token = (
            f"v{self.CACHE_VERSION}|k{self.kpt_sig}|n{len(self.keypoint_names)}|"
            f"b{backend}|m{m}"
        )
        return hashlib.sha1(token.encode("utf-8")).hexdigest()[:12]

    def _cache_dir(self) -> Path:
        return self.out_root / "posekit" / "predictions"

    def _cache_path(self, model_path: Path, backend: str) -> Path:
        return self._cache_dir() / f"{self._cache_key(model_path, backend)}.json"

    def _read_cache_file(
        self, path: Path, backend: str
    ) -> Optional[Dict[str, List[Tuple[float, float, float]]]]:
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            meta = data.get("meta", {})
            if int(meta.get("version", -1)) != self.CACHE_VERSION:
                return None
            if meta.get("backend") and str(meta.get("backend")) != backend:
                return None
            if "backend" not in meta and backend != "yolo":
                return None
            if meta.get("kpt_sig") != self.kpt_sig:
                return None
            if int(meta.get("num_kpts", -1)) != len(self.keypoint_names):
                return None
            preds = {}
            for k, v in (data.get("preds") or {}).items():
                preds[k] = [(float(x), float(y), float(c)) for x, y, c in v]
            return preds
        except Exception:
            return None

    def load_cache(
        self, model_path: Path, backend: str = "yolo"
    ) -> Dict[str, List[Tuple[float, float, float]]]:
        """load_cache method documentation."""
        key = self._cache_key(model_path, backend)
        if key in self._cache_mem:
            return self._cache_mem[key]
        preds = (
            self._read_cache_file(self._cache_path(model_path, backend), backend) or {}
        )
        self._cache_mem[key] = preds
        return preds

    def _write_cache(
        self,
        model_path: Path,
        preds: Dict[str, List[Tuple[float, float, float]]],
        backend: str = "yolo",
    ):
        self._cache_dir().mkdir(parents=True, exist_ok=True)
        meta = {
            "version": self.CACHE_VERSION,
            "kpt_sig": self.kpt_sig,
            "num_kpts": len(self.keypoint_names),
            "backend": backend,
            "model_sig": self._model_sig(model_path, backend),
        }
        payload = {"meta": meta, "preds": preds}
        path = self._cache_path(model_path, backend)
        path.write_text(json.dumps(payload), encoding="utf-8")
        self._cache_mem[self._cache_key(model_path, backend)] = preds

    def merge_cache(
        self: object,
        model_path: Path,
        new_preds: Dict[str, List[Tuple[float, float, float]]],
        backend: str = "yolo",
    ) -> object:
        """merge_cache method documentation."""
        cache = self.load_cache(model_path, backend)
        cache.update(new_preds)
        self._write_cache(model_path, cache, backend)

    def clear_cache(
        self, model_path: Optional[Path] = None, backend: str = "yolo"
    ) -> int:
        """clear_cache method documentation."""
        removed = 0
        if model_path is not None:
            key = self._cache_key(model_path, backend)
            if key in self._cache_mem:
                del self._cache_mem[key]
            try:
                path = self._cache_path(model_path, backend)
                if path.exists():
                    path.unlink()
                    removed += 1
            except Exception:
                pass
            return removed

        # Clear all caches
        self._cache_mem.clear()
        cache_dir = self._cache_dir()
        if not cache_dir.exists():
            return removed
        try:
            for p in cache_dir.glob("*.json"):
                try:
                    p.unlink()
                    removed += 1
                except Exception:
                    pass
        except Exception:
            pass
        return removed

    def get_cached_pred(
        self, model_path: Path, image_path: Path, backend: str = "yolo"
    ) -> Optional[List[Tuple[float, float, float]]]:
        """get_cached_pred method documentation."""
        preds = self.load_cache(model_path, backend)
        key = str(Path(image_path))
        if key in preds:
            return preds[key]
        key = str(Path(image_path).resolve())
        return preds.get(key)

    def get_cache_for_paths(
        self, model_path: Path, paths: List[Path], backend: str = "yolo"
    ) -> Optional[Dict[str, List[Tuple[float, float, float]]]]:
        """get_cache_for_paths method documentation."""
        preds = self.load_cache(model_path, backend)
        for p in paths:
            if str(p) not in preds and str(p.resolve()) not in preds:
                return None
        return preds

    def predict(
        self: object,
        model_path: Path,
        image_paths: List[Path | str],
        device: str,
        imgsz: int,
        conf: float,
        batch: int,
        progress_cb: object = None,
        cancel_cb: object = None,
        backend: str = "yolo",
        sleap_env: Optional[str] = None,
        sleap_device: str = "auto",
        sleap_batch: int = 4,
        sleap_max_instances: int = 1,
        sleap_runtime_flavor: str = "native",
        sleap_exported_model_path: Optional[str] = None,
        sleap_export_input_hw: Optional[Tuple[int, int]] = None,
        image_payloads: Optional[List[dict]] = None,
        cache_predictions: bool = True,
    ) -> Tuple[Optional[Dict[str, List[Tuple[float, float, float]]]], str]:
        """predict method documentation."""
        self._last_sleap_service_metrics = {}
        if not image_paths:
            return {}, ""
        if cancel_cb and cancel_cb():
            return None, "Canceled."

        backend = (backend or "yolo").lower()
        if backend == "sleap":
            if not model_path.exists() or not model_path.is_dir():
                return None, f"SLEAP model dir not found: {model_path}"
            sleap_env = str(sleap_env or "").strip()
            if not sleap_env:
                return None, "Select a SLEAP conda env."
            runtime_flavor = str(sleap_runtime_flavor or "native").strip().lower()
            # PoseKit is single-instance only.
            sleap_max_instances = 1
            tmp_dir = self.out_root / "posekit" / "tmp"
            tmp_dir.mkdir(parents=True, exist_ok=True)
            out_json = tmp_dir / f"sleap_pred_{os.getpid()}_{uuid.uuid4().hex}.json"
            exported_model_path = str(sleap_exported_model_path or "").strip()
            if runtime_flavor in {"onnx", "tensorrt"} and not exported_model_path:
                return (
                    None,
                    f"SLEAP {runtime_flavor} runtime requires an exported model path.",
                )
            service_metrics: Dict[str, float] = {}
            ok, preds, err = _run_sleap_predict_service(
                model_dir=model_path,
                image_paths=image_paths,
                image_payloads=image_payloads,
                out_json=out_json,
                keypoint_names=self.keypoint_names,
                skeleton_edges=self.skeleton_edges,
                env_name=sleap_env,
                device=sleap_device,
                batch=sleap_batch,
                max_instances=sleap_max_instances,
                runtime_flavor=runtime_flavor,
                exported_model_path=exported_model_path,
                export_input_hw=sleap_export_input_hw,
                progress_cb=progress_cb,
                cancel_cb=cancel_cb,
                metrics_out=service_metrics,
            )
            if not ok:
                return None, err
            self._last_sleap_service_metrics = service_metrics
            if cache_predictions:
                self.merge_cache(model_path, preds, backend=backend)
            return preds, ""

        if not model_path.exists() or not model_path.is_file():
            return None, f"Weights not found: {model_path}"
        if model_path.suffix != ".pt":
            return None, f"Invalid weights file: {model_path}"

        tmp_dir = self.out_root / "posekit" / "tmp"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        out_json = tmp_dir / f"pose_pred_{os.getpid()}_{uuid.uuid4().hex}.json"

        ok, preds, err = _run_pose_predict_subprocess(
            weights_path=model_path,
            image_paths=image_paths,
            out_json=out_json,
            device=device,
            imgsz=imgsz,
            conf=conf,
            batch=batch,
            progress_cb=progress_cb,
            cancel_cb=cancel_cb,
        )
        if not ok:
            return None, err

        if cache_predictions:
            self.merge_cache(model_path, preds, backend=backend)
        return preds, ""

    def get_last_sleap_service_metrics(self) -> Dict[str, float]:
        """Return timings from the most recent SLEAP service request."""
        return dict(self._last_sleap_service_metrics)

    @classmethod
    def shutdown_sleap_service(cls: object) -> object:
        """shutdown_sleap_service method documentation."""
        _get_sleap_service().stop()

    @classmethod
    def start_sleap_service(
        cls, env_name: str, out_root: Path
    ) -> Tuple[bool, str, Optional[Path]]:
        """start_sleap_service method documentation."""
        log_path = None
        try:
            log_dir = Path(out_root) / "posekit" / "logs"
            log_path = log_dir / f"sleap_service_{os.getpid()}_{uuid.uuid4().hex}.log"
        except Exception:
            log_path = None
        ok, err = _get_sleap_service().start(env_name, log_path=log_path)
        return ok, err, log_path

    @classmethod
    def sleap_service_running(cls) -> bool:
        """sleap_service_running method documentation."""
        svc = _get_sleap_service()
        return bool(svc.proc and svc.proc.poll() is None and svc.port)

    @classmethod
    def sleap_service_health(cls) -> Dict[str, object]:
        """Return health metadata reported by the persistent SLEAP service."""
        try:
            return _get_sleap_service().health()
        except Exception:
            return {}

    @classmethod
    def sleap_native_array_video_supported(cls) -> bool:
        """Return whether native SLEAP in the service env accepts in-memory arrays."""
        health = cls.sleap_service_health()
        return bool(health.get("native_array_video_supported", False))


def _run_pose_predict_subprocess(
    weights_path: Path,
    image_paths: List[Path],
    out_json: Path,
    device: str = "auto",
    imgsz: int = 640,
    conf: float = 0.25,
    batch: int = 16,
    progress_cb=None,
    cancel_cb=None,
) -> Tuple[bool, Dict[str, List[Tuple[float, float, float]]], str]:
    req = {
        "weights": str(weights_path),
        "images": [str(p) for p in image_paths],
        "device": device,
        "imgsz": int(imgsz),
        "conf": float(conf),
        "batch": int(batch),
        "out_json": str(out_json),
    }
    req_path = (
        Path(tempfile.gettempdir())
        / f"pose_pred_req_{os.getpid()}_{uuid.uuid4().hex}.json"
    )
    req_path.write_text(json.dumps(req), encoding="utf-8")

    code = (
        "import json,sys,os\n"
        "from pathlib import Path\n"
        "import numpy as np\n"
        "os.environ.setdefault('YOLO_AUTOINSTALL','false')\n"
        "os.environ.setdefault('ULTRALYTICS_SKIP_REQUIREMENTS_CHECKS','1')\n"
        "def _is_cuda_device(d):\n"
        "    d=(d or '').strip().lower()\n"
        "    return d in {'cuda','gpu'} or d.startswith('cuda:') or d.isdigit()\n"
        "def _maybe_limit_cuda_memory():\n"
        "    try:\n"
        "        import torch\n"
        "        if torch.cuda.is_available():\n"
        "            torch.cuda.set_per_process_memory_fraction(0.9)\n"
        "    except Exception:\n"
        "        pass\n"
        "req=Path(sys.argv[1])\n"
        "cfg=json.loads(req.read_text(encoding='utf-8'))\n"
        "from ultralytics import YOLO\n"
        "model=YOLO(cfg['weights'])\n"
        "pred_kwargs={\n"
        "  'imgsz': int(cfg['imgsz']),\n"
        "  'conf': float(cfg['conf']),\n"
        "  'max_det': 1,\n"
        "  'stream': True,\n"
        "  'verbose': False,\n"
        "  'batch': int(cfg['batch']),\n"
        "}\n"
        "if cfg.get('device') and cfg.get('device')!='auto':\n"
        "  pred_kwargs['device']=cfg['device']\n"
        "if _is_cuda_device(cfg.get('device')):\n"
        "  _maybe_limit_cuda_memory()\n"
        "  pred_kwargs['half']=True\n"
        "images=cfg['images']\n"
        "preds={}\n"
        "b=max(1,int(cfg.get('batch',1)))\n"
        "i=0\n"
        "total=len(images)\n"
        "done_total=0\n"
        "while i < len(images):\n"
        "  chunk=images[i:i+b]\n"
        "  kw=dict(pred_kwargs)\n"
        "  kw['source']=chunk\n"
        "  kw['batch']=min(b,len(chunk))\n"
        "  try:\n"
        "    results=list(model.predict(**kw))\n"
        "  except RuntimeError as e:\n"
        "    if 'out of memory' in str(e).lower() and b>1:\n"
        "      b=max(1,b//2)\n"
        "      continue\n"
        "    raise\n"
        "  for ri, r in enumerate(results):\n"
        "    path = None\n"
        "    if r is not None and getattr(r, 'path', None):\n"
        "      path = r.path\n"
        "    elif ri < len(chunk):\n"
        "      path = chunk[ri]\n"
        "    if path is None:\n"
        "      continue\n"
        "    if r is None or r.keypoints is None:\n"
        "      preds[str(Path(path))]=[]\n"
        "      done_total+=1\n"
        "      print(f'PROGRESS {done_total} {total}', flush=True)\n"
        "      continue\n"
        "    k=r.keypoints\n"
        "    try:\n"
        "      xy=k.xy\n"
        "      conf=getattr(k,'conf',None)\n"
        "      xy=xy.cpu().numpy() if hasattr(xy,'cpu') else np.array(xy)\n"
        "      if conf is not None:\n"
        "        conf=conf.cpu().numpy() if hasattr(conf,'cpu') else np.array(conf)\n"
        "    except Exception:\n"
        "      preds[str(Path(path))]=[]\n"
        "      done_total+=1\n"
        "      print(f'PROGRESS {done_total} {total}', flush=True)\n"
        "      continue\n"
        "    if xy.ndim==2:\n"
        "      xy=xy[None,:,:]\n"
        "    if conf is not None and conf.ndim==1:\n"
        "      conf=conf[None,:]\n"
        "    if xy.size==0:\n"
        "      preds[str(Path(path))]=[]\n"
        "      done_total+=1\n"
        "      print(f'PROGRESS {done_total} {total}', flush=True)\n"
        "      continue\n"
        "    if conf is not None:\n"
        "      scores=np.nanmean(conf,axis=1)\n"
        "    else:\n"
        "      scores=None\n"
        "      try:\n"
        "        if r.boxes is not None and hasattr(r.boxes,'conf'):\n"
        "          scores=r.boxes.conf.cpu().numpy()\n"
        "      except Exception:\n"
        "        scores=None\n"
        "      if scores is None:\n"
        "        scores=np.zeros((xy.shape[0],),dtype=np.float32)\n"
        "    best=int(np.argmax(scores)) if len(scores)>0 else 0\n"
        "    pred_xy=xy[best]\n"
        "    pred_conf=conf[best] if conf is not None else np.zeros((pred_xy.shape[0],),dtype=np.float32)\n"
        "    pred_conf=np.clip(np.asarray(pred_conf,dtype=np.float32),0.0,1.0)\n"
        "    pts=[]\n"
        "    for j in range(pred_xy.shape[0]):\n"
        "      c=float(pred_conf[j]) if j < len(pred_conf) else 0.0\n"
        "      pts.append((float(pred_xy[j][0]),float(pred_xy[j][1]),c))\n"
        "    preds[str(Path(path))]=pts\n"
        "    done_total+=1\n"
        "    print(f'PROGRESS {done_total} {total}', flush=True)\n"
        "  i+=len(chunk)\n"
        "out={'preds':preds}\n"
        "Path(cfg['out_json']).write_text(json.dumps(out),encoding='utf-8')\n"
    )

    popen_env = os.environ.copy()
    popen_env.setdefault("YOLO_AUTOINSTALL", "false")
    popen_env.setdefault("ULTRALYTICS_SKIP_REQUIREMENTS_CHECKS", "1")

    proc = subprocess.Popen(
        [sys.executable, "-c", code, str(req_path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=popen_env,
    )
    assert proc.stdout is not None
    last_msg = ""
    for line in proc.stdout:
        if cancel_cb and cancel_cb():
            try:
                proc.terminate()
            except Exception:
                pass
            return False, {}, "Canceled."
        line = line.strip()
        if line.startswith("PROGRESS "):
            try:
                _, done_s, total_s = line.split()
                if progress_cb:
                    progress_cb(int(done_s), int(total_s))
            except Exception:
                pass
        else:
            last_msg = line
    rc = proc.wait()
    if rc != 0:
        return False, {}, (last_msg or "Subprocess failed.")
    try:
        data = json.loads(out_json.read_text(encoding="utf-8"))
        return True, data.get("preds", {}), ""
    except Exception as e:
        return False, {}, str(e)


_SLEAP_SERVICE_CODE = textwrap.dedent(r"""
import base64,json,os,sys,threading,traceback,shutil,inspect,gc,subprocess,tempfile,time,uuid
from http.server import BaseHTTPRequestHandler, HTTPServer
from multiprocessing import shared_memory
from pathlib import Path
import numpy as np

try:
    import sleap_io as sio
except Exception as e:
    print(f'ERROR: sleap_io not installed: {e}', flush=True)
    sys.exit(2)

try:
    import cv2
except Exception:
    cv2 = None
try:
    from PIL import Image
except Exception:
    Image = None

_state = {
    'model_dir': None,
    'device': None,
    'predictor': None,
    'runtime_flavor': None,
    'exported_model_path': None,
    'batch': None,
    'max_instances': None,
    'export_input_hw': None,
    'export_input_channels': None,
    'preprocess_model_dir': None,
    'preprocess_config': None,
}
_log_path = None

# Per-request gc.collect() on the (large, torch-backed) heap is expensive and the
# HTTPServer is single-threaded, so every /infer call blocks on the previous
# call's full-heap collection. Under per-frame streaming pose (~1 call/frame)
# this dominated the gpu-tier pose cost (~46 ms/call, ~24 s over a 500-frame
# clip -- 4x the actual inference). Non-cyclic torch garbage is reclaimed by
# refcounting immediately; gc.collect() only reclaims reference cycles, so a
# periodic sweep bounds any cyclic growth without paying the cost every call.
# Cadence is env-tunable; set HYDRA_SLEAP_GC_EVERY=1 to restore per-request GC.
try:
    _GC_EVERY = max(1, int(os.environ.get('HYDRA_SLEAP_GC_EVERY', '50')))
except Exception:
    _GC_EVERY = 50
_req_count = 0

def _maybe_gc():
    global _req_count
    _req_count += 1
    if _req_count % _GC_EVERY == 0:
        gc.collect()

def _log(msg):
    try:
        if _log_path:
            with open(_log_path, "a", encoding="utf-8") as f:
                f.write(str(msg) + "\n")
    except Exception:
        pass
    try:
        print(msg, flush=True)
    except Exception:
        pass

def _normalize_device(device):
    d = (device or "").strip().lower()
    if d == "mps":
        try:
            import torch
            mps_backend = getattr(torch.backends, "mps", None)
            is_built = mps_backend.is_built() if mps_backend and hasattr(mps_backend, "is_built") else None
            is_avail = mps_backend.is_available() if mps_backend and hasattr(mps_backend, "is_available") else None
            _log(f"device mps requested (built={is_built} available={is_avail})")
        except Exception as e:
            _log(f"device mps requested (torch check failed: {e})")
        return "mps"
    return device

def _clear_predictor():
    _state['predictor']=None
    _state['model_dir']=None
    _state['runtime_flavor']=None
    _state['exported_model_path']=None
    _state['batch']=None
    _state['max_instances']=None
    _state['export_input_hw']=None
    _state['export_input_channels']=None
    _state['preprocess_model_dir']=None
    _state['preprocess_config']=None
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

def _make_skeleton(names, edges):
    try:
        return sio.Skeleton(nodes=names, edges=edges, name='PoseKit')
    except Exception:
        try:
            return sio.Skeleton(nodes=names, edges=edges)
        except Exception:
            if hasattr(sio,'Node') and hasattr(sio,'Skeleton'):
                nodes=[sio.Node(n) for n in names]
                sk=sio.Skeleton(nodes=nodes, name='PoseKit')
                for a,b in edges:
                    try:
                        sk.add_edge(nodes[int(a)], nodes[int(b)])
                    except Exception:
                        pass
                return sk
            raise

def _make_video_from_arrays(image_arrays):
    frames = [np.asarray(arr, dtype=np.uint8) for arr in (image_arrays or [])]
    if not frames:
        return None

    stacked = None
    if all(arr.shape == frames[0].shape for arr in frames):
        try:
            stacked = np.stack(frames, axis=0)
        except Exception:
            stacked = None

    load_video = getattr(sio, 'load_video', None)
    if callable(load_video):
        candidates = []
        if stacked is not None:
            candidates.append(stacked)
        candidates.append(frames)
        for source in candidates:
            try:
                video = load_video(source)
                if video is not None:
                    return video
            except Exception:
                pass

    if hasattr(sio, 'Video'):
        V = sio.Video
        for attr in ('from_numpy', 'from_image_array', 'from_images', 'from_frames'):
            factory = getattr(V, attr, None)
            if not callable(factory):
                continue
            candidates = []
            if stacked is not None:
                candidates.append((stacked, {}))
            candidates.append((frames, {}))
            if attr == 'from_numpy' and stacked is not None:
                candidates.extend(
                    [
                        (stacked, {'channels_last': True}),
                        (stacked, {'channels_first': False}),
                        (stacked, {'input_format': 'channels_last'}),
                    ]
                )
            for source, kwargs in candidates:
                try:
                    video = factory(source, **kwargs)
                    if video is not None:
                        return video
                except Exception:
                    pass

    return None

def _detect_native_array_video_support():
    cached = _state.get('native_array_video_supported', None)
    if cached is not None:
        return bool(cached)
    try:
        probe = [np.zeros((8, 8, 3), dtype=np.uint8) for _ in range(2)]
        supported = _make_video_from_arrays(probe) is not None
    except Exception:
        supported = False
    _state['native_array_video_supported'] = bool(supported)
    return bool(supported)

def _make_video(paths, image_arrays=None):
    if image_arrays is not None:
        video = _make_video_from_arrays(image_arrays)
        if video is not None:
            return video
    if hasattr(sio,'load_video'):
        try:
            return sio.load_video(paths)
        except Exception:
            pass
    if hasattr(sio,'Video'):
        V=sio.Video
        if hasattr(V,'from_image_files'):
            return V.from_image_files(paths)
        if hasattr(V,'from_filename') and len(paths)==1:
            return V.from_filename(paths[0])
    raise RuntimeError('Unable to create Video from image list')

def _make_labeled_frames(video, n):
    frames = []
    try:
        LF = getattr(sio, "LabeledFrame", None)
        if LF is None:
            _log("LabeledFrame not found in sleap_io")
            return frames
        for i in range(n):
            try:
                frames.append(LF(video=video, frame_idx=i, instances=[]))
            except Exception:
                frames.append(LF(video=video, frame_idx=i))
    except Exception:
        _log("LabeledFrame creation failed")
        return frames
    return frames

def _load_training_preprocess_config(model_dir):
    if _state.get('preprocess_model_dir') == model_dir:
        return _state.get('preprocess_config')
    preprocess_config = None
    cfg_path = Path(str(model_dir)) / 'training_config.yaml'
    if cfg_path.exists():
        try:
            import yaml
            raw = yaml.safe_load(cfg_path.read_text(encoding='utf-8')) or {}
            data_cfg = raw.get('data_config') or {}
            preprocess_config = data_cfg.get('preprocessing') or None
        except Exception as e:
            _log(f'predictor: failed to load preprocess_config from training_config.yaml: {e}')
    _state['preprocess_model_dir'] = model_dir
    _state['preprocess_config'] = preprocess_config
    return preprocess_config

def _load_predictor(model_dir, device):
    if _state.get('predictor') and _state.get('model_dir')==model_dir and _state.get('device')==device:
        return _state.get('predictor')
    _clear_predictor()
    pred = None
    preprocess_config = _load_training_preprocess_config(model_dir)
    try:
        from sleap_nn.inference import predictors as _pred_mod
        if hasattr(_pred_mod, 'Predictor'):
            predictor_cls = _pred_mod.Predictor
            if hasattr(predictor_cls, 'from_model_paths'):
                pred = predictor_cls.from_model_paths(
                    model_paths=[model_dir],
                    device=device,
                    preprocess_config=preprocess_config,
                )
            elif hasattr(predictor_cls, 'from_trained_models'):
                pred = predictor_cls.from_trained_models(
                    model_paths=[model_dir],
                    device=device,
                    preprocess_config=preprocess_config,
                )
    except Exception:
        pred = None
    if pred is None:
        try:
            import sleap_nn.predictors as _pred_mod2
            if hasattr(_pred_mod2, 'Predictor'):
                predictor_cls = _pred_mod2.Predictor
                if hasattr(predictor_cls, 'from_model_paths'):
                    pred = predictor_cls.from_model_paths(
                        model_paths=[model_dir],
                        device=device,
                        preprocess_config=preprocess_config,
                    )
                elif hasattr(predictor_cls, 'from_trained_models'):
                    pred = predictor_cls.from_trained_models(
                        model_paths=[model_dir],
                        device=device,
                        preprocess_config=preprocess_config,
                    )
        except Exception:
            pred = None
    if pred is not None:
        _state['predictor']=pred
        _state['model_dir']=model_dir
        _state['device']=device
        _log(f"predictor: loaded for {model_dir}")
    return pred

def _predict_with_predictor(pred, labels, batch, max_instances):
    queue_maxsize = 8
    try:
        if batch is not None:
            queue_maxsize = max(1, min(int(batch), 8))
    except Exception:
        queue_maxsize = 8
    try:
        if hasattr(pred, 'batch_size') and batch is not None:
            pred.batch_size = int(batch)
    except Exception:
        pass
    try:
        if hasattr(pred, 'max_instances') and max_instances is not None:
            pred.max_instances = int(max_instances)
    except Exception:
        pass
    if hasattr(pred, 'make_pipeline'):
        try:
            pred.make_pipeline(labels, queue_maxsize=queue_maxsize)
        except Exception:
            pass
    if hasattr(pred, 'predict'):
        fn = getattr(pred, 'predict')
        try:
            return fn(make_labels=True)
        except TypeError:
            try:
                return fn()
            except Exception:
                pass
    for meth in ('predict_labels','run'):
        if hasattr(pred, meth):
            fn=getattr(pred,meth)
            try:
                return fn(labels, batch_size=batch, max_instances=max_instances)
            except TypeError:
                try:
                    return fn(labels)
                except Exception:
                    pass
    return None

def _build_prepared_batch(pred, image_arrays):
    # Build a _run_inference_on_batch-ready batch directly from in-memory crop
    # arrays -- no Video, no make_pipeline, no reader thread. Replicates the
    # provider + _process_batch preprocessing (CHW, uint8, sizematcher, channel
    # handling). Crops arrive in cv2/BGR order (the temp-file path writes them via
    # cv2.imwrite, which sleap_io reads back as RGB), so swap BGR->RGB to match.
    import torch
    from sleap_nn.data.resizing import apply_sizematcher

    pc = pred.preprocess_config
    mh, mw = pc["max_height"], pc["max_width"]
    ensure_rgb = bool(pc.get("ensure_rgb", True))
    ensure_gray = bool(pc.get("ensure_grayscale", False))
    imgs, fidxs, vidxs, org_szs, eff_scales = [], [], [], [], []
    for i, arr in enumerate(image_arrays):
        a = np.ascontiguousarray(arr)
        if a.ndim == 2:
            a = a[:, :, None]
        if a.shape[2] == 3:
            a = a[:, :, ::-1]  # BGR -> RGB (match cv2.imwrite -> sio path)
        H, W = int(a.shape[0]), int(a.shape[1])
        chw = np.transpose(a, (2, 0, 1))
        t = torch.from_numpy(np.expand_dims(chw, 0).copy())  # (1, C, H, W) uint8
        t, eff = apply_sizematcher(t, mh, mw)
        if ensure_rgb and t.shape[-3] != 3:
            t = t.repeat(1, 3, 1, 1)
        elif ensure_gray and t.shape[-3] != 1:
            import torchvision.transforms.functional as _TF

            t = _TF.rgb_to_grayscale(t, num_output_channels=1)
        imgs.append(t.unsqueeze(0))  # (1, 1, C, mh, mw)
        org_szs.append(torch.Tensor([H, W]).unsqueeze(0).unsqueeze(0))  # (1, 1, 2)
        eff_scales.append(torch.tensor(eff))
        fidxs.append(i)
        vidxs.append(0)
    return imgs, fidxs, vidxs, org_szs, [], eff_scales


def _extract_preds_direct(raw, images, num_kpts):
    # Format raw _run_inference_on_batch output into the SAME structure as
    # _extract_preds: {str(Path(image)): [(x, y, score)] * num_kpts}. The peaks in
    # pred_instance_peaks are already final keypoint coordinates (sleap builds
    # PredictedInstances from them with no further transform).
    samples = []
    for ex in raw:
        pk = np.asarray(ex.get("pred_instance_peaks"))
        pv = ex.get("pred_peak_values")
        pv = np.asarray(pv) if pv is not None else None
        if pk.ndim == 2:  # single (K, 2) -> add batch dim
            pk = pk[None]
            pv = None if pv is None else pv[None]
        for b in range(pk.shape[0]):
            samples.append((pk[b], None if pv is None else pv[b]))
    preds = {}
    for i, path in enumerate(images):
        kp, vv = samples[i] if i < len(samples) else (None, None)
        pts = []
        for j in range(num_kpts):
            if kp is not None and j < kp.shape[0]:
                x, y = float(kp[j, 0]), float(kp[j, 1])
                c = float(vv[j]) if (vv is not None and j < len(vv)) else 1.0
            else:
                x, y, c = 0.0, 0.0, 0.0
            if not (np.isfinite(x) and np.isfinite(y)):
                pts.append((0.0, 0.0, 0.0))
            else:
                pts.append((x, y, c if np.isfinite(c) else 1.0))
        preds[str(Path(path))] = pts
    return preds


def _run_inference_direct(cfg, image_arrays, images, num_kpts):
    # Warm in-process SLEAP inference on in-memory crops: no Video, no
    # make_pipeline (reader thread), no CLI subprocess. ~1-6 ms/frame vs the CLI's
    # ~6 s. Returns preds in the _extract_preds format, or None to signal the
    # caller to fall back to the pipeline/CLI path.
    if image_arrays is None:
        return None
    device = _normalize_device(cfg.get("device"))
    pred = _load_predictor(cfg.get("model_dir"), device)
    if pred is None:
        return None
    if getattr(pred, "inference_model", None) is None:
        pred._initialize_inference_model()
    # make_pipeline normally sets this; _run_inference_on_batch requires it.
    pred.preprocess = True
    batch = _build_prepared_batch(pred, image_arrays)
    raw = list(pred._run_inference_on_batch(*batch))
    return _extract_preds_direct(raw, images, num_kpts)


def _run_inference(labels, model_dir, device, batch, max_instances):
    device = _normalize_device(device)
    pred = _load_predictor(model_dir, device)
    if pred is not None:
        _log("predictor: using python API")
        try:
            out = _predict_with_predictor(pred, labels, batch, max_instances)
            if out is not None:
                return out
        except Exception as e:
            _log(f"predictor: python API failed: {e}")
            if device == "mps":
                _log("predictor: retrying on cpu")
                return _run_inference(labels, model_dir, "cpu", batch, max_instances)
    try:
        from sleap_nn import predict as _predict_mod
        if hasattr(_predict_mod, 'run_inference'):
            fn=_predict_mod.run_inference
            _log("predictor: using sleap_nn.predict.run_inference")
            sig=inspect.signature(fn)
            kwargs={}
            if 'input_labels' in sig.parameters:
                kwargs['input_labels']=labels
            if 'model_paths' in sig.parameters:
                kwargs['model_paths']=[model_dir]
            if 'device' in sig.parameters and device and device!='auto':
                kwargs['device']=device
            if 'batch_size' in sig.parameters:
                kwargs['batch_size']=batch
            if 'max_instances' in sig.parameters:
                kwargs['max_instances']=max_instances
            try:
                return fn(**kwargs)
            except Exception as e:
                _log(f"predictor: run_inference failed: {e}")
                if device == "mps":
                    _log("predictor: retrying on cpu")
                    return _run_inference(labels, model_dir, "cpu", batch, max_instances)
    except Exception:
        pass
    return None

def _resolve_export_model_path(exported_model_path, runtime_flavor):
    p = Path(str(exported_model_path or "")).expanduser().resolve()
    if p.is_dir():
        if runtime_flavor == "onnx":
            files = sorted(p.rglob("*.onnx"))
            if not files:
                raise RuntimeError(f"No ONNX artifact found in export directory: {p}")
            return files[0]
        files = sorted(list(p.rglob("*.engine")) + list(p.rglob("*.trt")))
        if not files:
            raise RuntimeError(f"No TensorRT artifact found in export directory: {p}")
        return files[0]
    return p

def _extract_meta_value(meta, keys, default=None):
    for key in keys:
        if isinstance(meta, dict) and key in meta:
            return meta.get(key)
        if hasattr(meta, key):
            return getattr(meta, key)
    return default

def _detect_export_input_spec(exported_model_path):
    input_hw = None
    input_channels = None
    try:
        from sleap_nn.export.metadata import load_metadata
        meta = load_metadata(str(exported_model_path))
    except Exception:
        meta = None
    if meta is None:
        return None, None

    hw = _extract_meta_value(
        meta,
        ("crop_size", "crop_hw", "input_hw", "image_hw"),
        None,
    )
    if isinstance(hw, (list, tuple)) and len(hw) >= 2:
        try:
            input_hw = (int(hw[0]), int(hw[1]))
        except Exception:
            input_hw = None

    input_shape = _extract_meta_value(meta, ("input_image_shape", "input_shape"), None)
    if input_shape is not None:
        try:
            arr = np.asarray(input_shape).reshape((-1,))
            if arr.size >= 4:
                if int(arr[-1]) in (1, 3):
                    input_channels = int(arr[-1])
                    input_hw = (int(arr[-3]), int(arr[-2]))
                elif int(arr[1]) in (1, 3):
                    input_channels = int(arr[1])
                    input_hw = (int(arr[-2]), int(arr[-1]))
        except Exception:
            pass

    channels = _extract_meta_value(
        meta, ("input_channels", "channels", "num_channels"), None
    )
    if channels is not None:
        try:
            input_channels = int(channels)
        except Exception:
            pass
    return input_hw, input_channels

def _detect_predictor_input_spec(predictor):
    session = None
    for name in ("session", "_session", "ort_session", "_ort_session", "sess"):
        cand = getattr(predictor, name, None)
        if cand is not None and hasattr(cand, "get_inputs"):
            session = cand
            break
    if session is None or not hasattr(session, "get_inputs"):
        return None, None
    try:
        inputs = session.get_inputs()
    except Exception:
        return None, None
    if not inputs:
        return None, None
    try:
        shape = list(getattr(inputs[0], "shape", []) or [])
    except Exception:
        shape = []
    dims = []
    for d in shape:
        try:
            dims.append(int(d))
        except Exception:
            dims.append(-1)

    input_hw = None
    input_channels = None
    if len(dims) >= 4:
        if dims[-1] in (1, 3):
            input_channels = int(dims[-1])
            if dims[-3] > 0 and dims[-2] > 0:
                input_hw = (int(dims[-3]), int(dims[-2]))
        elif dims[1] in (1, 3):
            input_channels = int(dims[1])
            if dims[-2] > 0 and dims[-1] > 0:
                input_hw = (int(dims[-2]), int(dims[-1]))
    return input_hw, input_channels

def _detect_predictor_input_format(predictor):
    # Detect expected dtype and layout (NCHW/NHWC) from an ONNX session.
    session = None
    for attr in ("session", "_session", "ort_session", "_ort_session", "sess"):
        cand = getattr(predictor, attr, None)
        if cand is not None and hasattr(cand, "get_inputs"):
            session = cand
            break
    if session is None:
        return None
    try:
        inputs = session.get_inputs()
        if not inputs:
            return None
        inp = inputs[0]
        raw_type = str(getattr(inp, "type", "")).lower()
        shape = list(getattr(inp, "shape", []) or [])
    except Exception:
        return None
    is_float = "float" in raw_type
    dims = []
    for d in shape:
        try:
            dims.append(int(d))
        except (TypeError, ValueError):
            dims.append(-1)
    layout = "nhwc"
    if len(dims) >= 4:
        if dims[1] in (1, 3):
            layout = "nchw"
        elif dims[-1] in (1, 3):
            layout = "nhwc"
    return {"is_float": is_float, "layout": layout}

def _detect_model_min_batch(predictor):
    # Detect the minimum required batch from the model session/engine.
    # Returns the fixed batch size or None when the batch dim is dynamic.
    # ONNX session
    session = None
    for attr in ("session", "_session", "ort_session", "_ort_session", "sess"):
        cand = getattr(predictor, attr, None)
        if cand is not None and hasattr(cand, "get_inputs"):
            session = cand
            break
    if session is not None:
        try:
            inputs = session.get_inputs()
            if inputs:
                shape = getattr(inputs[0], "shape", [])
                if shape:
                    try:
                        b = int(shape[0])
                        if b > 0:
                            return b
                    except (TypeError, ValueError):
                        pass
        except Exception:
            pass
    # TRT engine binding / optimization profile
    for attr in ("engine", "_engine", "trt_engine"):
        engine = getattr(predictor, attr, None)
        if engine is None:
            continue
        if hasattr(engine, "get_profile_shape"):
            try:
                min_s, _, _ = engine.get_profile_shape(0, 0)
                if min_s and int(min_s[0]) > 0:
                    return int(min_s[0])
            except Exception:
                pass
        if hasattr(engine, "get_binding_shape"):
            try:
                shape = list(engine.get_binding_shape(0))
                if shape and int(shape[0]) > 0:
                    return int(shape[0])
            except Exception:
                pass
    return None

def _load_export_predictor(exported_model_path, runtime_flavor, device, batch, max_instances):
    runtime = str(runtime_flavor or "onnx").strip().lower()
    device = _normalize_device(device)
    export_file = _resolve_export_model_path(exported_model_path, runtime)
    export_key = str(export_file)
    batch = int(max(1, int(batch)))
    max_instances = int(max(1, int(max_instances)))
    if (
        _state.get("predictor") is not None
        and _state.get("runtime_flavor") == runtime
        and _state.get("exported_model_path") == export_key
        and _state.get("device") == device
        and int(_state.get("batch") or 0) == batch
        and int(_state.get("max_instances") or 0) == max_instances
    ):
        return _state.get("predictor")

    _clear_predictor()
    from sleap_nn.export.predictors import load_exported_model

    kwargs_base = {
        "batch_size": batch,
        "max_instances": max_instances,
    }
    if runtime == "onnx":
        if str(device).startswith("cuda"):
            kwargs_base["providers"] = ["CUDAExecutionProvider"]
        else:
            kwargs_base["providers"] = ["CPUExecutionProvider"]
    elif runtime == "tensorrt":
        if not str(device).startswith("cuda"):
            raise RuntimeError(
                f"SLEAP TensorRT runtime requires CUDA device, got: {device}"
            )
        kwargs_base["device"] = str(device)
    else:
        raise RuntimeError(f"Unsupported exported runtime: {runtime}")

    attempts = [
        {"runtime": runtime, **kwargs_base},
        {"inference_model": runtime, **kwargs_base},
        {"model_type": runtime, **kwargs_base},
        {"runtime": runtime},
        {},
    ]
    pred = None
    last_err = None
    for kwargs in attempts:
        try:
            pred = load_exported_model(str(export_file), **kwargs)
            break
        except Exception as exc:
            last_err = exc
            continue
    if pred is None:
        raise RuntimeError(
            f"Failed to initialize SLEAP exported predictor: {last_err}"
        )
    _state["predictor"] = pred
    _state["model_dir"] = None
    _state["device"] = device
    _state["runtime_flavor"] = runtime
    _state["exported_model_path"] = export_key
    _state["batch"] = batch
    _state["max_instances"] = max_instances
    in_hw, in_ch = _detect_export_input_spec(export_file)
    if in_hw is None:
        in_hw2, in_ch2 = _detect_predictor_input_spec(pred)
        in_hw = in_hw2
        if in_ch2 is not None:
            in_ch = in_ch2
    _state["export_input_hw"] = in_hw
    _state["export_input_channels"] = in_ch
    fmt = _detect_predictor_input_format(pred)
    _state["input_format"] = fmt
    min_b = _detect_model_min_batch(pred)
    _state["model_min_batch"] = min_b
    if fmt:
        _log(f"predictor: detected input format layout={fmt.get('layout')} is_float={fmt.get('is_float')}")
    if min_b:
        _log(f"predictor: detected model min/fixed batch={min_b}")
    _log(f"predictor: loaded exported {runtime} model {export_file}")
    return pred

def _load_image(path_str):
    p = str(path_str)
    if cv2 is not None:
        img = cv2.imread(p, cv2.IMREAD_COLOR)
        if img is not None:
            return img
    if Image is not None:
        with Image.open(p) as im:
            rgb = im.convert("RGB")
            arr = np.asarray(rgb)
            return arr[:, :, ::-1].copy()
    raise RuntimeError(
        "Unable to load image. Install either opencv-python or pillow in the SLEAP env."
    )

def _decode_image_payload(payload):
    if not isinstance(payload, dict):
        raise RuntimeError("Invalid in-memory SLEAP image payload.")
    image_id = str(payload.get("id") or "").strip()
    if not image_id:
        raise RuntimeError("In-memory SLEAP image payload is missing an id.")
    shape = payload.get("shape")
    if not isinstance(shape, (list, tuple)) or len(shape) not in (2, 3):
        raise RuntimeError(f"Invalid image shape for payload {image_id}: {shape}")
    try:
        shape = tuple(int(v) for v in shape)
    except Exception as exc:
        raise RuntimeError(f"Invalid image shape for payload {image_id}: {exc}")
    if any(dim <= 0 for dim in shape):
        raise RuntimeError(f"Non-positive image shape for payload {image_id}: {shape}")
    dtype_name = str(payload.get("dtype") or "uint8").strip().lower()
    if dtype_name != "uint8":
        raise RuntimeError(
            f"Unsupported image dtype for payload {image_id}: {dtype_name}"
        )
    expected = int(np.prod(shape))
    shm_name = str(payload.get("shm_name") or "").strip()
    if shm_name:
        shm = None
        try:
            shm = shared_memory.SharedMemory(name=shm_name)
            if int(getattr(shm, "size", expected)) < expected:
                raise RuntimeError(
                    f"Shared-memory payload {image_id} size mismatch: got {getattr(shm, 'size', 0)}, expected at least {expected}"
                )
            arr = np.ndarray(shape, dtype=np.uint8, buffer=shm.buf).copy()
        except Exception as exc:
            raise RuntimeError(f"Failed opening shared-memory payload {image_id}: {exc}")
        finally:
            if shm is not None:
                try:
                    shm.close()
                except Exception:
                    pass
    else:
        data_b64 = payload.get("data_b64")
        if not isinstance(data_b64, str) or not data_b64:
            raise RuntimeError(f"Payload {image_id} is missing image data.")
        try:
            raw = base64.b64decode(data_b64.encode("ascii"), validate=True)
        except Exception as exc:
            raise RuntimeError(f"Failed decoding payload {image_id}: {exc}")
        arr = np.frombuffer(raw, dtype=np.uint8)
        if arr.size != expected:
            raise RuntimeError(
                f"Payload {image_id} size mismatch: got {arr.size}, expected {expected}"
            )
        arr = arr.reshape(shape).copy()
    if arr.ndim == 3 and arr.shape[2] == 4:
        arr = arr[:, :, :3]
    return image_id, arr

def _decode_image_payloads(payloads):
    image_ids = []
    image_arrays = []
    for payload in payloads or []:
        image_id, arr = _decode_image_payload(payload)
        image_ids.append(image_id)
        image_arrays.append(arr)
    return image_ids, image_arrays

def _prepare_export_crop(crop, input_hw, input_channels):
    arr = np.asarray(crop, dtype=np.uint8)
    if arr.ndim == 2:
        arr = arr[:, :, None]
    if arr.ndim != 3:
        raise RuntimeError(f"Invalid crop shape for SLEAP export predictor: {arr.shape}")
    if arr.shape[2] > 3:
        arr = arr[:, :, :3]
    orig_h, orig_w = int(arr.shape[0]), int(arr.shape[1])

    if input_channels == 1 and arr.shape[2] != 1:
        if cv2 is not None:
            arr = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)[:, :, None]
        else:
            arr = np.mean(arr[:, :, :3], axis=2, keepdims=True).astype(arr.dtype)
    elif input_channels is None or input_channels == 3:
        if arr.shape[2] == 1:
            arr = np.repeat(arr, 3, axis=2)
        elif arr.shape[2] == 3:
            arr = arr[:, :, ::-1].copy()

    transform = (1.0, 0.0, 0.0, orig_w, orig_h)

    if input_hw is not None:
        h, w = int(input_hw[0]), int(input_hw[1])
        if h > 0 and w > 0 and (orig_h != h or orig_w != w):
            if cv2 is None:
                raise RuntimeError("OpenCV is required for resized SLEAP exported inference.")
            scale = min(float(w) / float(orig_w), float(h) / float(orig_h))
            new_w = max(1, int(round(float(orig_w) * scale)))
            new_h = max(1, int(round(float(orig_h) * scale)))
            pad_x = int((w - new_w) // 2)
            pad_y = int((h - new_h) // 2)
            if arr.shape[2] == 1:
                resized = cv2.resize(
                    arr[:, :, 0], (new_w, new_h), interpolation=cv2.INTER_LINEAR
                )
                canvas = np.zeros((h, w, 1), dtype=np.uint8)
                canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w, 0] = resized
                arr = canvas
            else:
                resized = cv2.resize(arr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                canvas = np.zeros((h, w, arr.shape[2]), dtype=np.uint8)
                canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized
                arr = canvas
            transform = (float(scale), float(pad_x), float(pad_y), orig_w, orig_h)

    return np.asarray(arr, dtype=np.uint8), transform

def _restore_export_keypoints(arr, transform):
    arr = np.asarray(arr, dtype=np.float32).copy()
    if arr.ndim != 2 or arr.shape[1] < 2:
        return arr
    scale, pad_x, pad_y, orig_w, orig_h = transform
    scale = float(scale) if float(scale) > 0.0 else 1.0
    arr[:, 0] = (arr[:, 0] - float(pad_x)) / scale
    arr[:, 1] = (arr[:, 1] - float(pad_y)) / scale
    arr[:, 0] = np.clip(arr[:, 0], 0.0, float(max(0, int(orig_w) - 1)))
    arr[:, 1] = np.clip(arr[:, 1], 0.0, float(max(0, int(orig_h) - 1)))
    return arr

def _predict_export_batch(predictor, crops, runtime_flavor):
    # Ensure all crops are 3D [H,W,C] for reliable stacking.
    crops_3d = []
    for c in crops:
        a = np.asarray(c, dtype=np.uint8)
        if a.ndim == 2:
            a = a[:, :, None]
        crops_3d.append(a)

    batch_uint8 = None
    try:
        batch_uint8 = np.stack(crops_3d, axis=0)  # [B, H, W, C]
    except Exception:
        batch_uint8 = None

    # Detected format from ONNX session (stored at predictor load time).
    fmt = _state.get("input_format")

    # If we detected the model's expected format, try it first (no guessing).
    if fmt is not None and batch_uint8 is not None:
        inp = batch_uint8.astype(np.float32) / 255.0 if fmt.get("is_float", True) else batch_uint8.copy()
        if fmt.get("layout") == "nchw" and inp.ndim == 4:
            inp = np.transpose(inp, (0, 3, 1, 2))
        try:
            return predictor.predict(inp)
        except Exception:
            _log(f"predictor: detected format failed ({fmt}), trying alternatives")

    # Fallback: try numpy array formats first, list(crops) last.
    attempts = []
    if batch_uint8 is not None:
        if batch_uint8.ndim == 4:
            attempts.append(np.transpose(batch_uint8.astype(np.float32) / 255.0, (0, 3, 1, 2)))  # NCHW float
        attempts.append(batch_uint8.astype(np.float32) / 255.0)  # NHWC float
        attempts.append(batch_uint8)  # NHWC uint8
    attempts.append(list(crops))  # raw list – last resort

    last_err = None
    for inp in attempts:
        try:
            return predictor.predict(inp)
        except Exception as exc:
            last_err = exc
            continue
    raise RuntimeError(
        f"SLEAP exported predictor failed for runtime '{runtime_flavor}': {last_err}"
    )

def _as_array(value):
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        return value
    if hasattr(value, "numpy"):
        try:
            return value.numpy()
        except Exception:
            pass
    if hasattr(value, "cpu"):
        try:
            return value.cpu().numpy()
        except Exception:
            pass
    try:
        return np.asarray(value)
    except Exception:
        return None

def _dict_first_present(mapping, keys):
    for key in keys:
        if key in mapping:
            return mapping.get(key)
    return None

def _normalize_conf_values(conf):
    arr = np.asarray(conf, dtype=np.float32)
    if arr.size == 0:
        return arr
    arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0)
    if np.any((arr < 0.0) | (arr > 1.0)):
        # Exported predictors can emit logits instead of probabilities.
        arr = 1.0 / (1.0 + np.exp(-np.clip(arr, -40.0, 40.0)))
    return np.clip(arr, 0.0, 1.0).astype(np.float32, copy=False)

def _pick_best_instance(xy, conf):
    if xy.ndim == 2 and xy.shape[1] == 2:
        n_kpts = int(xy.shape[0])
        if conf is None:
            conf_vec = np.zeros((n_kpts,), dtype=np.float32)
        else:
            conf_vec = _normalize_conf_values(conf).reshape((-1,))
            if conf_vec.size < n_kpts:
                conf_vec = np.pad(conf_vec, (0, n_kpts - conf_vec.size))
            elif conf_vec.size > n_kpts:
                conf_vec = conf_vec[:n_kpts]
        return np.column_stack((xy.astype(np.float32), conf_vec))

    if xy.ndim == 3 and xy.shape[-1] == 2:
        n_instances = int(xy.shape[0])
        n_kpts = int(xy.shape[1])
        if n_instances <= 0:
            return None
        if conf is None:
            conf = np.zeros((n_instances, n_kpts), dtype=np.float32)
        else:
            conf = _normalize_conf_values(conf)
            if conf.ndim == 1:
                conf = np.tile(conf[None, :], (n_instances, 1))
            elif conf.ndim > 2:
                conf = conf.reshape((n_instances, -1))
            if conf.shape[1] < n_kpts:
                pad = np.zeros((n_instances, n_kpts - conf.shape[1]), dtype=np.float32)
                conf = np.concatenate((conf, pad), axis=1)
            elif conf.shape[1] > n_kpts:
                conf = conf[:, :n_kpts]
        mean_scores = np.nanmean(conf, axis=1)
        idx = int(np.nanargmax(mean_scores)) if len(mean_scores) else 0
        xy_i = np.asarray(xy[idx], dtype=np.float32)
        conf_i = _normalize_conf_values(conf[idx]).reshape((-1,))
        if conf_i.size < n_kpts:
            conf_i = np.pad(conf_i, (0, n_kpts - conf_i.size))
        elif conf_i.size > n_kpts:
            conf_i = conf_i[:n_kpts]
        return np.column_stack((xy_i, conf_i))
    return None

def _coerce_prediction_batch(pred_out, batch_size):
    empty = [None] * int(max(0, batch_size))
    if pred_out is None or batch_size <= 0:
        return empty

    if isinstance(pred_out, dict):
        xy = _as_array(
            _dict_first_present(
                pred_out,
                ["instance_peaks", "pred_instance_peaks", "peaks", "keypoints", "points", "xy"],
            )
        )
        conf = _as_array(
            _dict_first_present(
                pred_out,
                [
                    "instance_peak_vals",
                    "pred_instance_peak_vals",
                    "pred_peak_vals",
                    "peak_vals",
                    "scores",
                    "confidences",
                    "confidence",
                ],
            )
        )
        if xy is None:
            return empty

        if xy.ndim == 2 and xy.shape[1] >= 2:
            xy = xy[None, :, :2]
            if conf is not None and conf.ndim == 1:
                conf = conf[None, :]
        elif xy.ndim == 3 and xy.shape[-1] >= 2:
            xy = xy[..., :2]
            # Could be [B,K,2] or [I,K,2] for a single sample.
            # If leading dim mismatches requested batch, interpret as instance-major.
            if int(xy.shape[0]) != int(batch_size):
                xy = xy[None, :, :, :]
                if conf is not None and conf.ndim == 2:
                    conf = conf[None, :, :]
        elif xy.ndim == 4 and xy.shape[-1] >= 2:
            xy = xy[..., :2]
        else:
            return empty

        out = []
        if xy.ndim == 3:
            for b in range(min(batch_size, xy.shape[0])):
                conf_b = conf[b] if conf is not None and conf.ndim >= 2 else None
                out.append(_pick_best_instance(xy[b], conf_b))
        else:
            for b in range(min(batch_size, xy.shape[0])):
                conf_b = conf[b] if conf is not None and conf.ndim >= 3 else None
                out.append(_pick_best_instance(xy[b], conf_b))
        if len(out) < batch_size:
            out.extend([None] * (batch_size - len(out)))
        return out[:batch_size]

    if isinstance(pred_out, (list, tuple)):
        # Handle canonical [xy, conf] pair.
        if len(pred_out) >= 1:
            xy0 = _as_array(pred_out[0])
            conf0 = _as_array(pred_out[1]) if len(pred_out) > 1 else None
            if xy0 is not None and isinstance(xy0, np.ndarray):
                if xy0.ndim >= 3 and xy0.shape[-1] >= 2:
                    return _coerce_prediction_batch(
                        {"instance_peaks": xy0, "instance_peak_vals": conf0},
                        batch_size,
                    )

        out = []
        for item in list(pred_out)[:batch_size]:
            if isinstance(item, dict):
                parsed = _coerce_prediction_batch(item, 1)
                out.append(parsed[0] if parsed else None)
                continue
            arr = _as_array(item)
            if arr is None:
                out.append(None)
                continue
            if arr.ndim == 2 and arr.shape[1] >= 3:
                out.append(np.asarray(arr[:, :3], dtype=np.float32))
            elif arr.ndim == 2 and arr.shape[1] >= 2:
                conf = np.zeros((arr.shape[0],), dtype=np.float32)
                out.append(np.column_stack((arr[:, :2].astype(np.float32), conf)))
            else:
                out.append(None)
        if len(out) < batch_size:
            out.extend([None] * (batch_size - len(out)))
        return out[:batch_size]

    arr = _as_array(pred_out)
    if arr is None:
        return empty
    if arr.ndim == 3 and arr.shape[-1] >= 3:
        out = [
            np.asarray(arr[i, :, :3], dtype=np.float32)
            for i in range(min(batch_size, arr.shape[0]))
        ]
        if len(out) < batch_size:
            out.extend([None] * (batch_size - len(out)))
        return out[:batch_size]
    if arr.ndim == 2 and arr.shape[1] >= 3 and batch_size == 1:
        return [np.asarray(arr[:, :3], dtype=np.float32)]
    return empty

def _normalize_export_xy_conf(raw, batch_size):
    parsed = _coerce_prediction_batch(raw, int(max(0, batch_size)))
    if not parsed:
        return None, None
    max_k = max(
        (
            int(arr.shape[0])
            for arr in parsed
            if arr is not None and hasattr(arr, "shape") and len(arr.shape) == 2
        ),
        default=0,
    )
    if max_k <= 0:
        return None, None

    xy_arr = np.zeros((batch_size, max_k, 2), dtype=np.float32)
    conf_arr = np.zeros((batch_size, max_k), dtype=np.float32)
    any_valid = False
    for i in range(min(batch_size, len(parsed))):
        arr = parsed[i]
        if arr is None:
            continue
        arr = np.asarray(arr, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[1] < 2:
            continue
        n = min(max_k, int(arr.shape[0]))
        if n <= 0:
            continue
        xy_arr[i, :n, :] = arr[:n, :2]
        if arr.shape[1] >= 3:
            conf_arr[i, :n] = _normalize_conf_values(arr[:n, 2])[:n]
        any_valid = True

    if not any_valid:
        return None, None
    return xy_arr, conf_arr

def _run_export_inference(cfg, images, num_kpts, image_arrays=None):
    runtime = str(cfg.get("runtime_flavor", "onnx")).strip().lower()
    exported_model_path = str(cfg.get("exported_model_path", "")).strip()
    if not exported_model_path:
        raise RuntimeError(f"SLEAP {runtime} runtime requires an exported model path.")

    predictor = _load_export_predictor(
        exported_model_path,
        runtime,
        cfg.get("device"),
        int(cfg.get("batch", 4)),
        int(cfg.get("max_instances", 1)),
    )
    input_hw = _state.get("export_input_hw")
    input_channels = _state.get("export_input_channels")

    forced_hw = cfg.get("export_input_hw")
    if isinstance(forced_hw, (list, tuple)) and len(forced_hw) >= 2:
        try:
            h = int(forced_hw[0])
            w = int(forced_hw[1])
            if h > 0 and w > 0:
                input_hw = (h, w)
        except Exception:
            pass
    if input_hw is None:
        raise RuntimeError(
            "SLEAP exported runtime requires fixed input shape metadata or an explicit export_input_hw. "
            "Re-export the model with fixed input height/width."
        )

    infer_batch = int(max(1, int(cfg.get("batch", 4))))
    model_min_batch = _state.get("model_min_batch")
    if model_min_batch is not None and int(model_min_batch) > infer_batch:
        _log(f"Model requires min batch={model_min_batch}, adjusting from {infer_batch}")
        infer_batch = int(model_min_batch)
    images_padded = list(images)
    raw_crops = list(image_arrays) if image_arrays is not None else None
    if images_padded and len(images_padded) < infer_batch:
        images_padded.extend([images_padded[-1]] * (infer_batch - len(images_padded)))
        if raw_crops is not None:
            raw_crops.extend([raw_crops[-1]] * (infer_batch - len(raw_crops)))

    if raw_crops is None:
        raw_crops = [_load_image(p) for p in images_padded]
    crops = []
    transforms = []
    for c in raw_crops:
        c2, transform = _prepare_export_crop(c, input_hw, input_channels)
        crops.append(c2)
        transforms.append(transform)
    raw = _predict_export_batch(predictor, crops, runtime)
    xy_arr, conf_arr = _normalize_export_xy_conf(raw, batch_size=len(images_padded))

    preds = {}
    for j, path in enumerate(images):
        rows = []
        if xy_arr is not None and j < xy_arr.shape[0]:
            xy = xy_arr[j]
            conf = (
                conf_arr[j]
                if conf_arr is not None and j < conf_arr.shape[0]
                else np.zeros((xy.shape[0],), dtype=np.float32)
            )
            n = min(int(xy.shape[0]), int(num_kpts))
            pts = np.zeros((n, 3), dtype=np.float32)
            if n > 0:
                pts[:, :2] = xy[:n, :2]
                pts[:, 2] = conf[:n]
                pts = _restore_export_keypoints(
                    pts,
                    transforms[j] if j < len(transforms) else (1.0, 0.0, 0.0, 0, 0),
                )
            for k in range(n):
                rows.append(
                    (
                        float(pts[k, 0]),
                        float(pts[k, 1]),
                        float(np.clip(pts[k, 2], 0.0, 1.0)),
                    )
                )
        if len(rows) < int(num_kpts):
            rows.extend([(0.0, 0.0, 0.0)] * (int(num_kpts) - len(rows)))
        preds[str(Path(path))] = rows
    return preds

def _predict_via_cli(cfg, labels):
    tmp_dir = Path(cfg['tmp_dir'])
    tmp_dir.mkdir(parents=True, exist_ok=True)
    input_slp = tmp_dir / 'input.slp'
    pred_slp = tmp_dir / 'pred.slp'
    if hasattr(sio,'save_file'):
        sio.save_file(labels, input_slp, format='slp')
    else:
        sio.save_slp(labels, input_slp)
    if shutil.which('sleap-nn') is None:
        raise RuntimeError('sleap-nn not found in PATH')
    cmd=['sleap-nn','track','--data_path',str(input_slp),'--model_paths',cfg['model_dir'],
         '--output_path',str(pred_slp)]
    device = _normalize_device(cfg.get('device'))
    if device and device!='auto':
        cmd += ['--device', str(device)]
    if cfg.get('batch'):
        cmd += ['--batch_size', str(int(cfg['batch']))]
    if cfg.get('max_instances'):
        cmd += ['--max_instances', str(int(cfg['max_instances']))]
    res=subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if res.returncode!=0:
        msg=(res.stdout or '').strip() or 'sleap-nn track failed'
        if device == "mps":
            _log(f"sleap-nn: track failed on mps, retrying cpu: {msg}")
            cmd = ['sleap-nn','track','--data_path',str(input_slp),'--model_paths',cfg['model_dir'],
                   '--output_path',str(pred_slp),'--device','cpu']
            if cfg.get('batch'):
                cmd += ['--batch_size', str(int(cfg['batch']))]
            if cfg.get('max_instances'):
                cmd += ['--max_instances', str(int(cfg['max_instances']))]
            res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            if res.returncode==0:
                _log('sleap-nn: track done (cpu fallback)')
                return sio.load_file(pred_slp)
            msg=(res.stdout or '').strip() or 'sleap-nn track failed'
        raise RuntimeError(msg)
    _log('sleap-nn: track done')
    return sio.load_file(pred_slp)

def _labels_from_output(out):
    if out is None:
        return None
    if hasattr(out,'labeled_frames') or hasattr(out,'predicted_frames') or hasattr(out,'frames'):
        return out
    if isinstance(out,(list,tuple)):
        for item in out:
            if hasattr(item,'labeled_frames') or hasattr(item,'predicted_frames') or hasattr(item,'frames'):
                return item
    return None

def _labels_summary(labels):
    if labels is None:
        return "labels=None"
    parts=[f"type={type(labels)}"]
    for name in ("labeled_frames","predicted_frames","frames"):
        if hasattr(labels, name):
            try:
                parts.append(f"{name}={len(getattr(labels,name) or [])}")
            except Exception:
                parts.append(f"{name}=?")
    if hasattr(labels, "instances"):
        try:
            parts.append(f"instances={len(getattr(labels,'instances') or [])}")
        except Exception:
            parts.append("instances=?")
    return " ".join(parts)

def _frame_attrs(lf):
    if lf is None:
        return "frame=None"
    parts=[]
    for name in ("predicted_instances","instances","predicted_instance","prediction","predictions","tracks"):
        if hasattr(lf, name):
            try:
                v=getattr(lf, name)
                if isinstance(v, (list,tuple)):
                    parts.append(f"{name}={len(v)}")
                elif v is None:
                    parts.append(f"{name}=None")
                else:
                    parts.append(f"{name}=1")
            except Exception:
                parts.append(f"{name}=?")
    return " ".join(parts) if parts else "no_instance_attrs"

def _labels_first_frame(labels):
    for name in ("labeled_frames","predicted_frames","frames"):
        if hasattr(labels, name):
            try:
                frames = getattr(labels, name) or []
                if frames:
                    return frames[0]
            except Exception:
                pass
    try:
        return labels[0]
    except Exception:
        return None

def _frame_index(lf, default_idx):
    for key in ('frame_idx','frame_index','idx'):
        if hasattr(lf, key):
            try:
                return int(getattr(lf, key))
            except Exception:
                pass
    return int(default_idx)

def _extract_preds(labels, images, num_kpts):
    frames = getattr(labels,'labeled_frames',[]) or []
    if not frames:
        frames = getattr(labels,'predicted_frames',[]) or []
    if not frames:
        frames = getattr(labels,'frames',[]) or []
    if not frames and hasattr(labels,'__len__') and hasattr(labels,'__getitem__'):
        tmp=[]
        for i in range(len(images)):
            try:
                tmp.append(labels[i])
            except Exception:
                pass
        frames = tmp
    by_idx={}
    for idx, lf in enumerate(frames):
        try:
            by_idx[_frame_index(lf, idx)]=lf
        except Exception:
            pass
    preds={}
    for i, path in enumerate(images):
        lf=by_idx.get(i)
        insts=[]
        if lf is not None:
            if getattr(lf,'predicted_instances',None) is not None:
                insts=list(lf.predicted_instances)
            elif getattr(lf,'instances',None) is not None:
                insts=list(lf.instances)
            elif getattr(lf,'predicted_instance',None) is not None:
                insts=[lf.predicted_instance]
            elif getattr(lf,'predictions',None) is not None:
                insts=list(lf.predictions)
            elif getattr(lf,'prediction',None) is not None:
                insts=[lf.prediction]
        if not insts:
            preds[str(Path(path))]=[]
            continue
        best=None
        best_score=-1e9
        for inst in insts:
            score=None
            if hasattr(inst,'score') and inst.score is not None:
                try:
                    score=float(inst.score)
                except Exception:
                    score=None
            if score is None:
                try:
                    arr=inst.numpy(scores=True)
                    if arr is not None and arr.shape[1] >= 3:
                        score=float(np.nanmean(arr[:,2]))
                except Exception:
                    score=None
            if score is None:
                score=0.0
            if best is None or score > best_score:
                best=inst
                best_score=score
        pts=[]
        if best is not None:
            arr=None
            try:
                arr=best.numpy(scores=True)
            except Exception:
                arr=None
            if arr is None:
                try:
                    arr=best.numpy()
                except Exception:
                    arr=None
            if arr is None:
                pts=[(0.0,0.0,0.0) for _ in range(num_kpts)]
            else:
                if arr.shape[1] == 2:
                    arr=np.concatenate([arr, np.zeros((arr.shape[0],1))], axis=1)
                if arr.shape[0] < num_kpts:
                    pad=np.zeros((num_kpts-arr.shape[0],3))
                    arr=np.concatenate([arr, pad], axis=0)
                if arr.shape[0] > num_kpts:
                    arr=arr[:num_kpts]
                for row in arr:
                    x,y,c = float(row[0]), float(row[1]), float(row[2])
                    if not np.isfinite(x) or not np.isfinite(y):
                        pts.append((0.0,0.0,0.0))
                    else:
                        if not np.isfinite(c):
                            c=1.0
                        pts.append((x,y,c))
        preds[str(Path(path))]=pts
    if not any(len(v)>0 for v in preds.values()):
        inst_list = getattr(labels, 'instances', None)
        if inst_list is not None:
            by_idx = {}
            for inst in list(inst_list):
                idx = None
                for key in ('frame_idx','frame_index','idx'):
                    if hasattr(inst, key):
                        try:
                            idx = int(getattr(inst, key))
                            break
                        except Exception:
                            pass
                if idx is None:
                    continue
                by_idx.setdefault(idx, []).append(inst)
            preds = {}
            for i, path in enumerate(images):
                insts = by_idx.get(i, [])
                if not insts:
                    preds[str(Path(path))]=[]
                    continue
                best = insts[0]
                best_score = -1e9
                for inst in insts:
                    score = getattr(inst, 'score', None)
                    try:
                        score = float(score) if score is not None else 0.0
                    except Exception:
                        score = 0.0
                    if score > best_score:
                        best = inst
                        best_score = score
                pts=[]
                try:
                    arr=best.numpy(scores=True)
                except Exception:
                    arr=None
                if arr is None:
                    pts=[(0.0,0.0,0.0) for _ in range(num_kpts)]
                else:
                    if arr.shape[1] == 2:
                        arr=np.concatenate([arr, np.zeros((arr.shape[0],1))], axis=1)
                    if arr.shape[0] < num_kpts:
                        pad=np.zeros((num_kpts-arr.shape[0],3))
                        arr=np.concatenate([arr, pad], axis=0)
                    if arr.shape[0] > num_kpts:
                        arr=arr[:num_kpts]
                    for row in arr:
                        x,y,c = float(row[0]), float(row[1]), float(row[2])
                        if not np.isfinite(x) or not np.isfinite(y):
                            pts.append((0.0,0.0,0.0))
                        else:
                            if not np.isfinite(c):
                                c=1.0
                            pts.append((x,y,c))
                preds[str(Path(path))]=pts
    return preds

class Handler(BaseHTTPRequestHandler):
    def _json(self, code, payload):
        body=json.dumps(payload).encode('utf-8')
        self.send_response(code)
        self.send_header('Content-Type','application/json')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)
    def do_GET(self):
        if self.path.startswith('/health'):
            self._json(
                200,
                {
                    'ok': True,
                    'model_dir': _state.get('model_dir'),
                    'native_array_video_supported': _detect_native_array_video_support(),
                },
            )
        else:
            self._json(404, {'ok': False, 'error': 'Not found'})
    def do_POST(self):
        if self.path.startswith('/shutdown'):
            self._json(200, {'ok': True})
            def _stop():
                threading.Thread(target=self.server.shutdown, daemon=True).start()
            _stop()
            return
        if not self.path.startswith('/infer'):
            self._json(404, {'ok': False, 'error': 'Not found'})
            return
        try:
            length=int(self.headers.get('Content-Length','0'))
            data=self.rfile.read(length).decode('utf-8') if length>0 else '{}'
            cfg=json.loads(data)
            image_payloads = cfg.get('image_payloads') or []
            image_arrays = None
            image_keys = [str(p) for p in (cfg.get('images') or [])]
            timings = {}
            if image_payloads:
                _decode_start = time.perf_counter()
                payload_ids, image_arrays = _decode_image_payloads(image_payloads)
                timings['service_decode_s'] = time.perf_counter() - _decode_start
                if image_keys:
                    if len(image_keys) != len(payload_ids):
                        raise RuntimeError('SLEAP image payload count mismatch.')
                else:
                    image_keys = payload_ids
            try:
                _log(f"INFER {len(image_keys)} model={cfg.get('model_dir')} device={cfg.get('device')}")
            except Exception:
                pass
            images=image_keys
            names=cfg.get('keypoint_names') or []
            edges=cfg.get('skeleton_edges') or []
            cfg['device'] = _normalize_device(cfg.get('device'))
            runtime_flavor = str(cfg.get("runtime_flavor", "native")).strip().lower()
            if runtime_flavor in ("onnx", "tensorrt"):
                try:
                    _export_start = time.perf_counter()
                    preds = _run_export_inference(
                        cfg,
                        images,
                        max(1, len(names)),
                        image_arrays=image_arrays,
                    )
                    timings['service_inference_s'] = time.perf_counter() - _export_start
                    try:
                        nonempty = sum(1 for v in preds.values() if v)
                        _log(f"preds_nonempty={nonempty}/{len(images)} runtime={runtime_flavor}")
                    except Exception:
                        pass
                    self._json(200, {'ok': True, 'preds': preds, 'timings_s': timings})
                    return
                except Exception as export_exc:
                    _log(
                        f"exported-runtime failed ({runtime_flavor}): {export_exc}; "
                        "falling back to native SLEAP runtime"
                    )
            # FAST PATH (native runtime, in-memory crops): warm predictor +
            # _run_inference_on_batch, no Video / make_pipeline / CLI. ~1-6 ms/frame
            # vs the CLI's ~6 s. Falls through to the pipeline/CLI path on any error
            # or empty result, so correctness never regresses.
            if runtime_flavor not in ("onnx", "tensorrt"):
                # Use the in-memory arrays if the client sent them (shared-memory
                # transport); otherwise read the temp-file PNGs into arrays so the
                # direct path applies regardless of transport. cv2.imread returns
                # BGR, matching what _build_prepared_batch expects.
                _direct_arrays = image_arrays
                if _direct_arrays is None:
                    try:
                        import cv2 as _cv2

                        _direct_arrays = []
                        for _p in images:
                            _im = _cv2.imread(str(_p))
                            if _im is None:
                                _direct_arrays = None
                                break
                            _direct_arrays.append(_im)
                    except Exception:
                        _direct_arrays = None
            else:
                _direct_arrays = None
            if _direct_arrays is not None:
                try:
                    _direct_start = time.perf_counter()
                    direct_preds = _run_inference_direct(
                        cfg, _direct_arrays, images, max(1, len(names))
                    )
                    timings['service_inference_s'] = timings.get(
                        'service_inference_s', 0.0
                    ) + (time.perf_counter() - _direct_start)
                    if direct_preds is not None and any(
                        len(v) > 0 for v in direct_preds.values()
                    ):
                        _log(
                            f"preds_nonempty="
                            f"{sum(1 for v in direct_preds.values() if v)}/{len(images)}"
                            f" runtime=direct"
                        )
                        self._json(200, {'ok': True, 'preds': direct_preds, 'timings_s': timings})
                        return
                    _log("direct path: no preds; falling back to pipeline/CLI")
                except Exception as _direct_exc:
                    _log(f"direct path failed ({_direct_exc}); falling back to pipeline/CLI")
            sk=_make_skeleton(names, edges)
            request_tmp_dir = None
            native_images = images
            if image_arrays is not None:
                # Pose golden rule: build the video from the in-memory arrays
                # only. If construction fails, fail loud and let it propagate
                # rather than materializing PNGs to disk.
                _video_start = time.perf_counter()
                video = _make_video(native_images, image_arrays=image_arrays)
                timings['service_materialize_s'] = timings.get('service_materialize_s', 0.0) + (time.perf_counter() - _video_start)
            else:
                _video_start = time.perf_counter()
                video=_make_video(native_images)
                timings['service_materialize_s'] = timings.get('service_materialize_s', 0.0) + (time.perf_counter() - _video_start)
            try:
                _log(f"video type={type(video)}")
            except Exception:
                pass
            frames = _make_labeled_frames(video, len(images))
            try:
                _log(f"frames_created={len(frames)}")
            except Exception:
                pass
            try:
                labels=sio.Labels(videos=[video], skeletons=[sk], labeled_frames=frames)
            except Exception:
                labels=sio.Labels(videos=[video], skeletons=[sk])
            _infer_start = time.perf_counter()
            out=_run_inference(labels, cfg.get('model_dir'), cfg.get('device'), int(cfg.get('batch',4)), int(cfg.get('max_instances',1)))
            timings['service_inference_s'] = timings.get('service_inference_s', 0.0) + (time.perf_counter() - _infer_start)
            labels_out=_labels_from_output(out)
            used_cli=False
            if labels_out is None:
                _infer_start = time.perf_counter()
                labels_out=_predict_via_cli(cfg, labels)
                timings['service_inference_s'] = timings.get('service_inference_s', 0.0) + (time.perf_counter() - _infer_start)
                used_cli=True
            _log(_labels_summary(labels_out))
            _log(f"frame0: {_frame_attrs(_labels_first_frame(labels_out))}")
            _extract_start = time.perf_counter()
            preds=_extract_preds(labels_out, images, len(names))
            timings['service_extract_s'] = timings.get('service_extract_s', 0.0) + (time.perf_counter() - _extract_start)
            if not preds or not any(len(v)>0 for v in preds.values()):
                if not used_cli:
                    _infer_start = time.perf_counter()
                    labels_out=_predict_via_cli(cfg, labels)
                    timings['service_inference_s'] = timings.get('service_inference_s', 0.0) + (time.perf_counter() - _infer_start)
                    _log("fallback: cli used")
                    _log(_labels_summary(labels_out))
                    _log(f"frame0: {_frame_attrs(_labels_first_frame(labels_out))}")
                    _extract_start = time.perf_counter()
                    preds=_extract_preds(labels_out, images, len(names))
                    timings['service_extract_s'] = timings.get('service_extract_s', 0.0) + (time.perf_counter() - _extract_start)
            try:
                nonempty=sum(1 for v in preds.values() if v)
                _log(f"preds_nonempty={nonempty}/{len(images)}")
            except Exception:
                pass
            self._json(200, {'ok': True, 'preds': preds, 'timings_s': timings})
        except Exception as e:
            _log(traceback.format_exc())
            self._json(500, {'ok': False, 'error': str(e)})
        finally:
            try:
                if 'request_tmp_dir' in locals() and request_tmp_dir is not None:
                    shutil.rmtree(request_tmp_dir, ignore_errors=True)
            except Exception:
                pass
            _maybe_gc()
    def log_message(self, format, *args):
        return

def main():
    cfg=Path(sys.argv[1])
    data=json.loads(cfg.read_text(encoding='utf-8'))
    global _log_path
    _log_path = data.get('log_path')
    host=data.get('host','127.0.0.1')
    port=int(data.get('port',0))
    httpd=HTTPServer((host, port), Handler)
    actual_port=httpd.server_address[1]
    _log(f'READY {host}:{actual_port}')
    httpd.serve_forever()

if __name__=='__main__':
    main()
""").strip()


class _SleapHttpService:
    def __init__(self):
        self.env_name: Optional[str] = None
        self.proc: Optional[subprocess.Popen] = None
        self.port: Optional[int] = None
        self._stdout_thread: Optional[threading.Thread] = None
        self._last_log: str = ""
        self.log_path: Optional[Path] = None

    def _drain_stdout(self):
        if not self.proc or not self.proc.stdout:
            return
        log_file = None
        if self.log_path:
            try:
                self.log_path.parent.mkdir(parents=True, exist_ok=True)
                log_file = open(self.log_path, "a", encoding="utf-8")
            except Exception:
                log_file = None
        for line in self.proc.stdout:
            if line:
                self._last_log = line.strip()
                if log_file:
                    try:
                        log_file.write(line)
                        log_file.flush()
                    except Exception:
                        pass
        if log_file:
            try:
                log_file.close()
            except Exception:
                pass

    def _pick_free_port(self) -> int:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
        sock.close()
        return port

    def _append_log(self, message: str) -> None:
        """Append a line to the log file if configured."""
        if not self.log_path:
            return
        try:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(message + "\n")
        except Exception:
            pass

    def _write_temp_files(
        self, port: int
    ) -> Tuple[Optional[Path], Optional[Path], str]:
        """Write service config and code to temp files.

        Returns (cfg_path, code_path, error_message).
        """
        cfg = {"host": "127.0.0.1", "port": int(port)}
        if self.log_path:
            cfg["log_path"] = str(self.log_path)
        cfg_path = (
            Path(tempfile.gettempdir())
            / f"sleap_service_{os.getpid()}_{uuid.uuid4().hex}.json"
        )
        cfg_path.write_text(json.dumps(cfg), encoding="utf-8")
        code_path = (
            Path(tempfile.gettempdir())
            / f"sleap_service_{os.getpid()}_{uuid.uuid4().hex}.py"
        )
        try:
            code_path.write_text(_SLEAP_SERVICE_CODE, encoding="utf-8")
        except Exception:
            return None, None, "Failed to write SLEAP service code."
        return cfg_path, code_path, ""

    def _wait_for_health(self, timeout: float = 15.0) -> Tuple[bool, str]:
        """Poll the service health endpoint until ready or timeout."""
        start = time.time()
        while time.time() - start < timeout:
            if self.proc.poll() is not None:
                err = self._last_log or "SLEAP service failed to start."
                if self.log_path:
                    err = f"{err} (log: {self.log_path})"
                return False, err
            try:
                url = f"http://127.0.0.1:{self.port}/health"
                with urllib.request.urlopen(url, timeout=0.5) as resp:
                    if resp.status == 200:
                        self._append_log("[health] ok")
                        return True, ""
            except Exception:
                time.sleep(0.2)
        self.stop()
        err = "Timed out starting SLEAP service."
        if self.log_path:
            err = f"{err} (log: {self.log_path})"
        return False, err

    def health(self, timeout: float = 2.0) -> dict:
        """Return the JSON payload from the service health endpoint."""
        if not self.port:
            raise RuntimeError("SLEAP service not running.")
        url = f"http://127.0.0.1:{self.port}/health"
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            body = resp.read().decode("utf-8")
            return json.loads(body) if body else {}

    def start(self, env_name: str, log_path: Optional[Path] = None) -> Tuple[bool, str]:
        """start method documentation."""
        if self.proc and self.proc.poll() is None and self.env_name == env_name:
            return True, ""
        self.stop()
        self.log_path = log_path
        if self.log_path:
            try:
                self.log_path.parent.mkdir(parents=True, exist_ok=True)
            except Exception:
                pass
            self._append_log(f"[start] env={env_name}")
        if shutil.which("conda") is None:
            return False, "Conda not found on PATH."
        preflight_ok, preflight_err = _sleap_env_preflight(env_name)
        if not preflight_ok:
            self._append_log(f"[preflight] failed: {preflight_err}")
            return False, preflight_err
        port = self._pick_free_port()
        cfg_path, code_path, err = self._write_temp_files(port)
        if err:
            return False, err
        self.proc = popen_conda(
            [
                "conda",
                "run",
                "-n",
                env_name,
                "python",
                "-u",
                str(code_path),
                str(cfg_path),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert self.proc.stdout is not None
        self._stdout_thread = threading.Thread(target=self._drain_stdout, daemon=True)
        self._stdout_thread.start()
        self.port = port
        self.env_name = env_name
        return self._wait_for_health()

    def stop(self: object) -> object:
        """stop method documentation."""
        if self.proc and self.proc.poll() is None and self.port:
            try:
                self.request("/shutdown", {}, timeout=2)
            except Exception:
                pass
        if self.proc and self.proc.poll() is None:
            try:
                self.proc.terminate()
            except Exception:
                pass
        if self.proc and self.proc.poll() is None:
            try:
                self.proc.wait(timeout=2)
            except Exception:
                try:
                    self.proc.kill()
                except Exception:
                    pass
        self.proc = None
        self.port = None
        self.env_name = None

    def request(self, path: str, payload: dict, timeout: float = 3600.0) -> dict:
        """request method documentation."""
        if not self.port:
            raise RuntimeError("SLEAP service not running.")
        url = f"http://127.0.0.1:{self.port}{path}"
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8")
            return json.loads(body) if body else {}


_SLEAP_SERVICE = _SleapHttpService()


def _get_sleap_service() -> _SleapHttpService:
    return _SLEAP_SERVICE


def _format_sleap_env_preflight_error(raw_error: str, env_name: str) -> str:
    msg = str(raw_error or "").strip()
    base = (
        f"SLEAP env preflight failed for conda env '{env_name}'. "
        "The environment cannot import SLEAP runtime modules."
    )
    if "undefined symbol: ncclAlltoAll" in msg or (
        "libtorch_cuda.so" in msg and "undefined symbol" in msg
    ):
        return (
            f"{base} Detected CUDA/NCCL binary mismatch while importing torch "
            "(libtorch_cuda.so unresolved nccl symbol).\n"
            "Recommended fix (choose one):\n"
            "  A) Use CPU-only torch in this env:\n"
            f"     conda run -n {env_name} python -m pip uninstall -y torch torchvision torchaudio nvidia-nccl-cu12 nvidia-nccl-cu13\n"
            f"     conda run -n {env_name} python -m pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision\n"
            "  B) Reinstall GPU torch/torchvision + NCCL from one matching CUDA stack (same channel/index):\n"
            f"     conda run -n {env_name} python -m pip uninstall -y torch torchvision torchaudio nvidia-nccl-cu12 nvidia-nccl-cu13\n"
            f"     conda run -n {env_name} python -m pip install --index-url https://download.pytorch.org/whl/cu130 torch torchvision\n"
            f"     conda run -n {env_name} python -m pip install --upgrade nvidia-nccl-cu13\n"
            f'  Verify: conda run -n {env_name} python -c "import torch, torchvision; print(torch.__version__, torchvision.__version__, torch.cuda.is_available())"\n'
            f"Original error: {msg}"
        )
    if "torchvision::nms does not exist" in msg:
        return (
            f"{base} Detected torchvision operator registration failure "
            "(operator torchvision::nms does not exist), which usually means "
            "torch and torchvision are incompatible builds.\n"
            "Recommended fix:\n"
            f"  conda run -n {env_name} python -m pip uninstall -y torch torchvision torchaudio\n"
            "  reinstall torch/torchvision from the same release channel (CPU or CUDA-specific)\n"
            f'  conda run -n {env_name} python -c "import torch, torchvision; print(torch.__version__, torchvision.__version__)"\n'
            f"Original error: {msg}"
        )
    return f"{base} Original error: {msg}"


def _sleap_env_preflight(env_name: str) -> Tuple[bool, str]:
    env_name = str(env_name or "").strip()
    if not env_name:
        return False, "Select a SLEAP conda env."
    script = (
        "import json,sys\n"
        "out={'python':sys.version.split()[0]}\n"
        "try:\n"
        "    import torch\n"
        "    out['torch']=getattr(torch,'__version__',None)\n"
        "except Exception as e:\n"
        "    print(f'ERROR: torch import failed: {e}')\n"
        "    sys.exit(2)\n"
        "try:\n"
        "    import torchvision\n"
        "    out['torchvision']=getattr(torchvision,'__version__',None)\n"
        "except Exception as e:\n"
        "    print(f'ERROR: torchvision import failed: {e}')\n"
        "    sys.exit(3)\n"
        "try:\n"
        "    from sleap_nn.inference import predictors as _p\n"
        "    out['sleap_predictors']=bool(_p is not None)\n"
        "except Exception as e:\n"
        "    print(f'ERROR: sleap_nn predictors import failed: {e}')\n"
        "    sys.exit(4)\n"
        "print(json.dumps(out))\n"
    )
    script_path = (
        Path(tempfile.gettempdir())
        / f"sleap_preflight_{os.getpid()}_{uuid.uuid4().hex}.py"
    )
    try:
        script_path.write_text(script, encoding="utf-8")
    except Exception as exc:
        return False, f"Failed to prepare SLEAP env preflight script: {exc}"

    try:
        res = run_conda(
            ["conda", "run", "-n", env_name, "python", str(script_path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=90,
        )
    except subprocess.TimeoutExpired:
        return False, (
            f"SLEAP env preflight timed out for conda env '{env_name}'. "
            "Try running: conda run -n "
            f'{env_name} python -c "import torch, torchvision, sleap_nn"'
        )
    except Exception as exc:
        return False, f"SLEAP env preflight execution failed for '{env_name}': {exc}"
    finally:
        try:
            if script_path.exists():
                script_path.unlink()
        except Exception:
            pass

    if int(res.returncode) == 0:
        return True, ""

    raw = (res.stdout or "").strip() or f"preflight exited with code {res.returncode}"
    return False, _format_sleap_env_preflight_error(raw, env_name)


def _all_zero_pose_chunk(
    chunk_paths: List[Path],
    chunk_preds: Dict[str, List[Tuple[float, float, float]]],
    num_keypoints: int,
) -> bool:
    if not chunk_paths or not chunk_preds:
        return False
    n_kpts = int(max(1, num_keypoints))
    for p in chunk_paths:
        k1 = str(Path(p))
        k2 = str(Path(p).resolve())
        rows = chunk_preds.get(k1)
        if rows is None:
            rows = chunk_preds.get(k2)
        if rows is None or len(rows) < n_kpts:
            return False
        for row in rows[:n_kpts]:
            if not isinstance(row, (list, tuple)) or len(row) < 3:
                return False
            try:
                x = float(row[0])
                y = float(row[1])
                c = float(row[2])
            except Exception:
                return False
            if abs(x) > 1e-9 or abs(y) > 1e-9 or abs(c) > 1e-9:
                return False
    return True


def _run_sleap_predict_service(
    model_dir: Path,
    image_paths: List[Path | str],
    out_json: Path,
    keypoint_names: List[str],
    skeleton_edges: List[Tuple[int, int]],
    env_name: str,
    device: str = "auto",
    batch: int = 4,
    max_instances: int = 1,
    runtime_flavor: str = "native",
    exported_model_path: str = "",
    export_input_hw: Optional[Tuple[int, int]] = None,
    progress_cb=None,
    cancel_cb=None,
    image_payloads: Optional[List[dict]] = None,
    metrics_out: Optional[Dict[str, float]] = None,
) -> Tuple[bool, Dict[str, List[Tuple[float, float, float]]], str]:
    svc = _get_sleap_service()
    log_path = None
    try:
        out_root = out_json.parents[2]
        log_dir = out_root / "posekit" / "logs"
        log_path = log_dir / f"sleap_service_{os.getpid()}_{uuid.uuid4().hex}.log"
    except Exception:
        log_path = None
    ok, err = svc.start(env_name, log_path=log_path)
    if not ok:
        return False, {}, err

    total = len(image_paths)
    preds: Dict[str, List[Tuple[float, float, float]]] = {}
    aggregated_metrics: Dict[str, float] = {}
    chunk_size = max(1, min(total, max(8, int(batch) * 8)))
    done = 0
    for i in range(0, total, chunk_size):
        if cancel_cb and cancel_cb():
            return False, {}, "Canceled."
        chunk = image_paths[i : i + chunk_size]
        chunk_payloads = None
        if image_payloads is not None:
            chunk_payloads = image_payloads[i : i + chunk_size]
        payload = {
            "images": [str(p) for p in chunk],
            "image_payloads": chunk_payloads,
            "model_dir": str(model_dir),
            "device": device,
            "batch": int(batch),
            "max_instances": int(max_instances),
            "keypoint_names": list(keypoint_names),
            "skeleton_edges": [list(e) for e in skeleton_edges],
            "tmp_dir": str(Path(tempfile.gettempdir()) / f"sleap_srv_{os.getpid()}"),
            "runtime_flavor": str(runtime_flavor or "native").strip().lower(),
            "exported_model_path": str(exported_model_path or "").strip(),
            "export_input_hw": (
                [int(export_input_hw[0]), int(export_input_hw[1])]
                if isinstance(export_input_hw, (tuple, list))
                and len(export_input_hw) >= 2
                and int(export_input_hw[0]) > 0
                and int(export_input_hw[1]) > 0
                else None
            ),
        }
        try:
            resp = svc.request("/infer", payload, timeout=3600.0)
        except urllib.error.HTTPError as e:
            try:
                body = e.read().decode("utf-8")
                err = body or f"HTTP error {e.code}"
                if svc.log_path:
                    err = f"{err} (log: {svc.log_path})"
                return False, {}, err
            except Exception:
                err = f"HTTP error {e.code}"
                if svc.log_path:
                    err = f"{err} (log: {svc.log_path})"
                return False, {}, err
        except Exception as e:
            err = str(e)
            if svc.log_path:
                err = f"{err} (log: {svc.log_path})"
            return False, {}, err
        if not resp or not resp.get("ok"):
            err = resp.get("error", "SLEAP service error.")
            if svc.log_path:
                err = f"{err} (log: {svc.log_path})"
            return False, {}, err
        for key, value in dict(resp.get("timings_s") or {}).items():
            try:
                key_str = str(key)
                aggregated_metrics[key_str] = aggregated_metrics.get(
                    key_str, 0.0
                ) + float(value)
            except Exception:
                continue
        chunk_preds = dict(resp.get("preds", {}) or {})
        runtime = str(runtime_flavor or "native").strip().lower()
        if (
            runtime in {"onnx", "tensorrt"}
            and str(exported_model_path or "").strip()
            and _all_zero_pose_chunk(chunk, chunk_preds, len(keypoint_names))
        ):
            fb_json = (
                Path(tempfile.gettempdir())
                / f"sleap_export_fb_{os.getpid()}_{uuid.uuid4().hex}.json"
            )
            fb_ok, fb_preds, _fb_err = _run_sleap_export_predict_subprocess(
                exported_model_path=Path(str(exported_model_path).strip()),
                runtime_flavor=runtime,
                image_paths=chunk,
                out_json=fb_json,
                env_name=env_name,
                keypoint_names=keypoint_names,
                device=device,
                batch=batch,
                max_instances=max_instances,
                input_hw=export_input_hw,
                progress_cb=None,
                cancel_cb=cancel_cb,
            )
            if fb_ok and fb_preds:
                chunk_preds = fb_preds
        preds.update(chunk_preds)
        done += len(chunk)
        if progress_cb:
            progress_cb(done, total)

    if metrics_out is not None:
        metrics_out.clear()
        metrics_out.update(aggregated_metrics)

    out_json.write_text(json.dumps({"preds": preds}), encoding="utf-8")
    return True, preds, ""


def _run_sleap_export_predict_subprocess(
    exported_model_path: Path,
    runtime_flavor: str,
    image_paths: List[Path],
    out_json: Path,
    env_name: str,
    keypoint_names: List[str],
    device: str = "auto",
    batch: int = 4,
    max_instances: int = 1,
    input_hw: Optional[Tuple[int, int]] = None,
    progress_cb=None,
    cancel_cb=None,
) -> Tuple[bool, Dict[str, List[Tuple[float, float, float]]], str]:
    env_name = str(env_name or "").strip()
    if not env_name:
        return False, {}, "Select a SLEAP conda env."
    if shutil.which("conda") is None:
        return False, {}, "Conda not found on PATH."

    worker_path = (Path(__file__).resolve().parent / "worker.py").resolve()
    if not worker_path.exists():
        return False, {}, f"SLEAP export worker not found: {worker_path}"

    req = {
        "exported_model_path": str(exported_model_path),
        "runtime_flavor": str(runtime_flavor or "onnx").strip().lower(),
        "images": [str(p) for p in image_paths],
        "out_json": str(out_json),
        "device": str(device or "auto"),
        "batch": int(batch),
        "max_instances": int(max_instances),
        "num_keypoints": int(max(1, len(keypoint_names))),
        "input_hw": (
            [int(input_hw[0]), int(input_hw[1])]
            if isinstance(input_hw, (tuple, list))
            and len(input_hw) >= 2
            and int(input_hw[0]) > 0
            and int(input_hw[1]) > 0
            else None
        ),
    }
    req_path = (
        Path(tempfile.gettempdir())
        / f"sleap_export_req_{os.getpid()}_{uuid.uuid4().hex}.json"
    )
    req_path.write_text(json.dumps(req), encoding="utf-8")

    proc = popen_conda(
        [
            "conda",
            "run",
            "-n",
            env_name,
            "python",
            str(worker_path),
            str(req_path),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert proc.stdout is not None
    tail_lines: List[str] = []
    for line in proc.stdout:
        if cancel_cb and cancel_cb():
            try:
                proc.terminate()
            except Exception:
                pass
            return False, {}, "Canceled."
        line = line.strip()
        if line.startswith("PROGRESS "):
            try:
                _, done_s, total_s = line.split()
                if progress_cb:
                    progress_cb(int(done_s), int(total_s))
            except Exception:
                pass
        else:
            if line:
                tail_lines.append(line)
                if len(tail_lines) > 20:
                    tail_lines.pop(0)

    rc = proc.wait()
    if rc != 0:
        msg = "\n".join(tail_lines[-12:]).strip()
        if not msg:
            msg = "SLEAP exported runtime subprocess failed."
        return False, {}, msg

    try:
        data = json.loads(out_json.read_text(encoding="utf-8"))
        return True, data.get("preds", {}), ""
    except Exception as e:
        return False, {}, str(e)
