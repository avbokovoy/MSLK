# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

"""Integration layer for ROCm/aiter kernels.

This module exposes three categories of aiter kernels through the standard
``torch.ops.mslk.*`` dispatch interface:

* **Python/Triton** — ``aiter_softmax`` (calls ``aiter`` as a runtime
  dependency via ``import aiter``; no source vendoring, no C++ build changes)
* **C++/HIP** — ``aiter_rmsnorm``, ``aiter_add_rmsnorm`` (compiled into
  mslk.so from aiter's source via the ``external/aiter`` submodule)
* **Assembly** — ``aiter_topk_softmax`` (pre-compiled HSACO code objects
  loaded at runtime via ``hipModuleLoad``)

All imports are gated on ``torch.version.hip`` so this module is safely
importable on CUDA-only builds (the ops simply won't be available).

Python/Triton integration pattern
----------------------------------
aiter exposes its Python ops at the top level (``aiter.rmsnorm``,
``aiter.fused_topk``, etc.) and Triton ops under ``aiter.ops.triton.*``.
MSLK wraps these behind ``torch.ops.mslk.*`` registration so they
participate in torch dispatch / torch.compile / Meta.  No source code is
copied — aiter must be installed as a pip package.  See the installation
instructions below.

Installation
------------
The ``amd-aiter`` wheel is published on GitHub Releases (not PyPI) with
ROCm-version-specific builds.  To install::

    # 1. Download the wheel matching your ROCm version from:
    #    https://github.com/ROCm/aiter/releases
    pip install amd_aiter-<version>+rocm<X.Y>...-<cpXY>-...-manylinux_2_28_x86_64.whl

    # 2. Or, from a source checkout with Triton-only (no C++ JIT):
    AITER_TRITON_ONLY=1 pip install -e external/aiter
"""

import logging
import warnings

import torch

_logger = logging.getLogger(__name__)

_AITER_INSTALL_MSG = (
    "aiter is not installed. aiter Python/Triton kernels will not be "
    "available in mslk.aiter. To install, download the wheel for your "
    "ROCm version from https://github.com/ROCm/aiter/releases and run:\n"
    "  pip install amd_aiter-<version>+rocm<X.Y>...-manylinux_2_28_x86_64.whl\n"
    "Or from the submodule:\n"
    "  AITER_TRITON_ONLY=1 pip install -e external/aiter"
)

from . import _meta  # noqa: F401  — registers Meta impls

if torch.version.hip is not None:
    # -----------------------------------------------------------------
    # Python/Triton kernels — imported from aiter as a runtime dependency.
    # -----------------------------------------------------------------
    try:
        from .softmax import softmax  # noqa: F401

        _logger.debug("aiter Python/Triton kernels loaded successfully")
    except ImportError:
        warnings.warn(_AITER_INSTALL_MSG, stacklevel=2)

    # C++/HIP and ASM wrappers — available only when mslk.so was built with
    # USE_ROCM and the op schemas were registered in aiter_ops.cpp.
    if hasattr(torch.ops.mslk, "aiter_rmsnorm"):
        from .rmsnorm import add_rmsnorm, rmsnorm  # noqa: F401

    if hasattr(torch.ops.mslk, "aiter_topk_softmax"):
        from .topk_softmax import topk_softmax  # noqa: F401
