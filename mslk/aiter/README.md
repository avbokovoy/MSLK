# aiter Integration

Integrates [ROCm/aiter](https://github.com/ROCm/aiter) kernels into MSLK. Three kernel categories are supported, each with a different integration mechanism.

## Kernel Categories

| Category | Example Op | Integration | Source |
|----------|-----------|-------------|--------|
| Python/Triton | `aiter_softmax` | Runtime `import aiter` | Installed `amd-aiter` package |
| C++/HIP | `aiter_rmsnorm` | Compiled into `mslk.so` | `external/aiter` submodule |
| Assembly | `aiter_topk_softmax` | `.co` loaded via `hipModuleLoad` | `external/aiter` submodule |

All ops are registered under `torch.ops.mslk.*` and gated on ROCm.

## Installation

### Python/Triton kernels

Download the wheel matching your ROCm version from [aiter releases](https://github.com/ROCm/aiter/releases):

```bash
pip install amd_aiter-0.1.22+rocm7.1...-cp312-...-manylinux_2_28_x86_64.whl
```

Or from the submodule (Triton ops only, no C++ JIT):

```bash
AITER_TRITON_ONLY=1 pip install -e external/aiter
```

### C++/HIP and Assembly kernels

These are compiled into `mslk.so` from the `external/aiter` submodule during the MSLK build. No separate install step is needed.

Requires:
- ROCm build (`MSLK_BUILD_VARIANT=rocm`)
- Submodule initialized: `git submodule update --init --recursive`

## Architecture

### Python/Triton (`softmax.py`)

```
aiter package (pip)  →  mslk.aiter.softmax  →  torch.ops.mslk.aiter_softmax
```

- Calls `aiter.ops.triton.softmax.softmax()` directly
- Registers as `torch.ops.mslk.aiter_softmax` via `torch.library`
- Optional: `try/except ImportError` with install instructions if aiter is missing

### C++/HIP (`rmsnorm`)

```
external/aiter/csrc/kernels/rmsnorm_quant_kernels.cu
    ↓ (compiled into mslk.so as HIP source)
csrc/aiter/aiter_ops.cpp  →  TORCH_LIBRARY_FRAGMENT(mslk, m)
    ↓ (at::Tensor → aiter_tensor_t bridge)
include/mslk/aiter/aiter_bridge.h
    ↓
torch.ops.mslk.aiter_rmsnorm
```

- `aiter_bridge.h` converts `at::Tensor` to `aiter_tensor_t` (aiter's torch-free POD)
- Op schemas and dispatch registered in `csrc/aiter/aiter_ops.cpp` under `#ifdef USE_ROCM`

### Assembly (`topk_softmax`)

```
external/aiter/hsa/gfx942/topksoftmax/*.co  (precompiled HSACO)
    ↓ (loaded at runtime via hipModuleLoad)
csrc/aiter/asm_topksoftmax.cu  →  AiterAsmKernel loader
    ↓ (config header generated at build time by hsa/codegen.py)
torch.ops.mslk.aiter_topk_softmax
```

- `.co` files are precompiled GPU assembly, shipped as package data
- `hsa/codegen.py` generates `asm_topksoftmax_configs.hpp` at CMake build time
- Heuristic kernel selection based on GPU arch, dtype, expert count, topk

## Build System

### CMakeLists.txt

- Adds `external/aiter/csrc/include` to include paths
- Runs `hsa/codegen.py` via `add_custom_command` to generate ASM config headers

### MslkDefault.cmake

- `csrc/aiter/*.cpp` added to CPU source glob
- `external/aiter/csrc/kernels/rmsnorm_quant_kernels.cu` and `csrc/aiter/asm_topksoftmax.cu` added as HIP sources (ROCm only)

### CI (`ci/scripts/utils_aiter.bash`)

- `install_aiter_from_submodule()` runs `AITER_TRITON_ONLY=1 pip install -e external/aiter`
- Called from `prepare_mslk_build()` in `ci/scripts/mslk_build.bash`
- Version pin: `ci/aiter_version.txt`, submodule pinned to matching tag

## Files

| Path | Purpose |
|------|---------|
| `mslk/aiter/__init__.py` | Module init, ROCm-gated imports, install warning |
| `mslk/aiter/softmax.py` | Triton softmax — delegates to `import aiter` |
| `mslk/aiter/rmsnorm.py` | Python wrapper for C++/HIP rmsnorm |
| `mslk/aiter/topk_softmax.py` | Python wrapper for ASM topk_softmax |
| `mslk/aiter/_meta.py` | Meta (shape inference) registrations |
| `csrc/aiter/aiter_ops.cpp` | C++ op schemas + ROCm dispatch |
| `csrc/aiter/asm_topksoftmax.cu` | ASM kernel wrapper |
| `include/mslk/aiter/aiter_bridge.h` | `at::Tensor` ↔ `aiter_tensor_t` bridge |
| `ci/aiter_version.txt` | Version pin |
| `ci/scripts/utils_aiter.bash` | CI install script |
