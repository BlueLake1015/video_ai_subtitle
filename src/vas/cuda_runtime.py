"""CUDA runtime library preloader.

CTranslate2 (faster-whisper) and several other GPU libraries `dlopen()` cuBLAS
and cuDNN at runtime. The PyPI wheels we depend on (`nvidia-cublas-cu12`,
`nvidia-cudnn-cu12`) install the .so files into `site-packages/nvidia/...`
but those paths are NOT on the linker's default search path, and we don't want
to require users to set LD_LIBRARY_PATH manually.

Calling `preload_cuda_libs()` resolves both:
    1. Locates the .so files in the installed nvidia-* wheels.
    2. dlopen()s them with RTLD_GLOBAL so subsequent loads (e.g. CTranslate2's
       own libctranslate2.so) find their symbols.

This is a no-op on non-Linux platforms and silently skips libraries that
aren't installed (e.g. CPU-only environments).
"""

from __future__ import annotations

import ctypes
import os
import sys
from importlib import import_module
from pathlib import Path


_PRELOADED = False


def preload_cuda_libs() -> None:
    """Preload libcublas + libcudnn. Idempotent; safe to call repeatedly."""
    global _PRELOADED
    if _PRELOADED or sys.platform != "linux":
        _PRELOADED = True
        return

    candidates = [
        # (pip-package-module, expected .so basename)
        ("nvidia.cublas.lib", "libcublas.so.12"),
        ("nvidia.cudnn.lib",  "libcudnn.so.9"),
        ("nvidia.cudnn.lib",  "libcudnn_ops.so.9"),
        ("nvidia.cudnn.lib",  "libcudnn_cnn.so.9"),
    ]

    for mod_name, soname in candidates:
        try:
            mod = import_module(mod_name)
        except ImportError:
            continue
        # nvidia.*.lib are namespace packages: __file__ may be None,
        # but __path__ always points at the directory.
        lib_dirs: list[Path] = []
        mod_file = getattr(mod, "__file__", None)
        if mod_file:
            lib_dirs.append(Path(mod_file).resolve().parent)
        for p in getattr(mod, "__path__", []) or []:
            lib_dirs.append(Path(p).resolve())
        for lib_dir in lib_dirs:
            so_path = lib_dir / soname
            if so_path.exists():
                try:
                    ctypes.CDLL(str(so_path), mode=ctypes.RTLD_GLOBAL)
                except OSError:
                    # Preload failed; the consumer's dlopen will produce a
                    # clearer error than we can here.
                    pass
                break

    _PRELOADED = True
