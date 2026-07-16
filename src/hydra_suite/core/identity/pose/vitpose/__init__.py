"""Native-PyTorch ViTPose. Standalone leaf: imports nothing from hydra_suite.

Ports ViTPose (github.com/ViTAE-Transformer/ViTPose) with no mmcv/mmpose
dependency. Upstream's backbone imports zero mmcv and no compiled op touches the
forward pass; the whole coupling is config/registry/runner scaffolding.

Module attribute names deliberately mirror the upstream checkpoint's state_dict
keys (`backbone.*`, `keypoint_head.*`, `blocks.{i}.attn.qkv`, `patch_embed.proj`,
`last_norm`), so load_state_dict(strict=True) needs no rename map.

Historical note on OpenMP: this package once imported cv2 before torch to dodge
`OMP: Error #15` from two libomp copies (conda's @rpath/libomp.dylib and torch's
vendored /opt/llvm-openmp/lib/libomp.dylib). That never actually worked -- isort
sorts third-party `import torch` ahead of first-party `from hydra_suite...`, so
torch always won the race anyway. The real fix was applied to the environment:
torch/lib/libomp.dylib is now a symlink to the conda libomp, so exactly one
OpenMP runtime maps and no import order matters. See
docs/superpowers/specs/2026-07-16-vitpose-backend-roadmap.md.
"""
