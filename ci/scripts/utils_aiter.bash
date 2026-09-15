#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


# shellcheck disable=SC1091,SC2128
. "$( dirname -- "$BASH_SOURCE"; )/utils_base.bash"

################################################################################
# aiter Setup Functions
################################################################################

install_aiter_from_submodule () {
  local env_name="$1"
  if [ "$env_name" == "" ]; then
    echo "Usage: ${FUNCNAME[0]} ENV_NAME"
    echo "Example(s):"
    echo "    ${FUNCNAME[0]} build_env"
    return 1
  else
    echo "################################################################################"
    echo "# Install aiter Python layer from submodule (editable, Triton-only)"
    echo "#"
    echo "# [$(date --utc +%FT%T.%3NZ)] + ${FUNCNAME[0]} ${*}"
    echo "################################################################################"
    echo ""
  fi

  # aiter is a ROCm-only dependency; skip it on other build variants.
  if [ "$BUILD_VARIANT" != "rocm" ]; then
    echo "[BUILD] Skipping aiter install for BUILD_VARIANT '${BUILD_VARIANT}' (rocm only)"
    return 0
  fi

  local aiter_submodule_dir="${GITHUB_WORKSPACE:-$(git rev-parse --show-toplevel)}/external/aiter"

  if [ ! -d "$aiter_submodule_dir" ]; then
    echo "[BUILD] WARNING: external/aiter submodule not found at ${aiter_submodule_dir}, skipping"
    return 0
  fi

  # Single source of truth for the aiter pin, shared with setup.py.
  # shellcheck disable=SC2155
  local expected_version=$(tr -d '[:space:]' < "$( dirname -- "$BASH_SOURCE"; )/../aiter_version.txt")

  # shellcheck disable=SC2155
  local env_prefix=$(env_name_or_prefix "${env_name}")

  # AITER_TRITON_ONLY=1 skips aiter's C++/HIP JIT compilation, installing
  # only the Python/Triton layer.  The C++ headers and .co assembly blobs
  # are consumed directly from the submodule tree by MSLK's CMake build.
  echo "[BUILD] Installing aiter Python layer (Triton-only) from ${aiter_submodule_dir} ..."
  # shellcheck disable=SC2086
  (exec_with_retries 3 conda run --no-capture-output ${env_prefix} \
    env AITER_TRITON_ONLY=1 python -m pip install --no-build-isolation -e "${aiter_submodule_dir}") || return 1

  # shellcheck disable=SC2086
  (test_python_import_package "${env_name}" aiter) || return 1

  echo "[CHECK] Printing out the aiter version ..."
  # shellcheck disable=SC2086,SC2155
  local installed_aiter_version=$(conda run ${env_prefix} python -c "import aiter; print(getattr(aiter, '__version__', 'unknown'))")
  echo "################################################################################"
  echo "[CHECK] The installed VERSION of aiter is: ${installed_aiter_version}"
  echo "[CHECK] The expected VERSION of aiter is: ${expected_version}"
  echo "################################################################################"
  echo ""
}
