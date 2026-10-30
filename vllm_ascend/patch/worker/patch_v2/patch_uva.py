# Adapt from https://github.com/vllm-project/vllm/blob/main/vllm/v1/worker/gpu/block_table.py
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
#
import os
from collections.abc import Callable, Sequence
from functools import cache
from importlib.util import find_spec
from pathlib import Path

import numpy as np
import torch
import vllm.v1.worker.gpu.buffer_utils
from vllm.logger import logger

from vllm_ascend.device.hardware_profile import HardwareCapability, get_current_hardware_profile

_TRITON_HOST_POINTER_REJECTION_MARKERS = (
    "attributes.location.type",
    "cannot be accessed from Triton",
)


def is_real_uva_requested() -> bool:
    """Return whether the allocator was configured for mapped pinned memory."""
    allocator_options = os.environ.get("PYTORCH_NPU_ALLOC_CONF", "").split(",")
    return any(option.strip() == "pinned_mem_register:True" for option in allocator_options)


def supports_real_uva_hardware() -> bool:
    """Return whether the active profile supports Triton mapped-host loads."""
    return get_current_hardware_profile().supports(HardwareCapability.TRITON_MAPPED_HOST_LOAD)


def _launcher_source_allows_mapped_host_pointers(source: str) -> bool:
    """Check for the host-pointer rejection removed by Triton-Ascend #1759."""
    return not all(marker in source for marker in _TRITON_HOST_POINTER_REJECTION_MARKERS)


@cache
def triton_launcher_allows_mapped_host_pointers() -> bool:
    """Return whether the installed launcher lacks the known UVA rejection.

    Version checks cannot distinguish the original 3.2.2 wheel from rebuilt
    packages containing Triton-Ascend #1759. Inspecting the installed launcher
    also works for vendor and daily builds that keep the same package version.
    When the source cannot be inspected, keep the explicit-copy fallback.
    """
    try:
        driver_spec = find_spec("triton.backends.ascend.driver")
    except (ImportError, AttributeError, ValueError):
        driver_spec = None

    if driver_spec is None or driver_spec.origin is None:
        logger.warning_once("Cannot inspect the Triton-Ascend launcher; using the async H2D fallback for UVA")
        return False

    try:
        source = Path(driver_spec.origin).read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        logger.warning_once("Cannot read the Triton-Ascend launcher; using the async H2D fallback for UVA")
        return False

    if not _launcher_source_allows_mapped_host_pointers(source):
        logger.warning_once(
            "The Triton-Ascend launcher rejects mapped host pointers; using the async H2D fallback for UVA. "
            "See https://github.com/triton-lang/triton-ascend/issues/783"
        )
        return False
    return True


def get_row_indices_from_key(key: int | slice | tuple, dim_size: int) -> set[int]:
    """get the set of row indices involved in the given key."""
    if isinstance(key, int):
        # parse index such as np[1]
        key = key if key >= 0 else dim_size + key
        # handle negative index
        if key < 0 or key >= dim_size:
            raise IndexError(f"row index {key} out of [0, {dim_size})")
        return {key}
    elif isinstance(key, slice):
        # parse slice such as np[1:3]
        start, stop, step = key.indices(dim_size)
        return set(range(start, stop, step))
    elif isinstance(key, tuple):
        # parse row slice such as np[1,:100]
        if len(key) == 0:
            return set(range(dim_size))
        return get_row_indices_from_key(key[0], dim_size)
    else:
        # for other types such as list/ndarray, we return all rows.
        return set(range(dim_size))


class MonitoredNumPyArray:
    """A wrapper around a NumPy array that monitors modifications."""

    def __init__(self, array: np.ndarray, callback: Callable):
        self._array = array
        self._callback = callback

    def __setitem__(self, key, value):
        self._array[key] = value
        dim_size = self._array.shape[0]
        row_indices = get_row_indices_from_key(key, dim_size)
        for row in row_indices:
            self._callback(row)

    def __getitem__(self, key):
        return self._array[key]

    def __getattr__(self, name):
        return getattr(self._array, name)


class MonitoredTorchTensor:
    """A wrapper around a torch tensor that monitors modifications."""

    def __init__(self, tensor: torch.Tensor, callback: Callable):
        self._tensor = tensor
        self._callback = callback

    def __setitem__(self, key, value):
        self._tensor[key] = value
        dim_size = self._tensor.size(0)
        row_indices = get_row_indices_from_key(key, dim_size)
        for row in row_indices:
            self._callback(row)

    def __getitem__(self, key):
        return self._tensor[key]

    def __getattr__(self, name):
        return getattr(self._tensor, name)


def _zeros_on_npu(cpu: torch.Tensor) -> torch.Tensor:
    """Zero an NPU buffer matching ``cpu``.

    ``aclnnInplaceZero`` rejects ``DT_UINT64``. Prompt-embeds pointer tables
    use that dtype, so zero an int64 allocation and view it back. The zero
    bit pattern is identical, and callers still see ``uint64``.
    """
    if cpu.dtype == torch.uint64:
        return torch.zeros(cpu.shape, dtype=torch.int64, device="npu").view(torch.uint64)
    return torch.zeros_like(cpu, device="npu")


def _int64_storage(tensor: torch.Tensor) -> torch.Tensor:
    """View a uint64 tensor as int64 so NPU copy kernels accept it."""
    if tensor.dtype == torch.uint64:
        return tensor.view(torch.int64)
    return tensor


class UvaBufferWrapper:
    """Provide mapped UVA views with an explicit-copy fallback on Ascend.

    Set ``PYTORCH_NPU_ALLOC_CONF=pinned_mem_register:True`` before process
    startup to request registered pinned allocations. The installed Triton
    launcher and the runtime mapping result decide whether the mapped view is
    safe to expose; otherwise the wrapper keeps using the copy fallback.
    """

    def __init__(self, size: int | Sequence[int], dtype: torch.dtype):
        self._cpu: torch.Tensor = torch.zeros(size, dtype=dtype, device="cpu", pin_memory=True)
        self._np: np.ndarray = self._cpu.numpy()
        self._modified_indices: set[int] = set()
        requested_real_uva = is_real_uva_requested()
        hardware_allows_real_uva = supports_real_uva_hardware()
        if requested_real_uva and not hardware_allows_real_uva:
            logger.warning_once(
                "The current Ascend hardware profile does not support Triton mapped-host loads; "
                "using the async H2D fallback for UVA"
            )
        launcher_allows_real_uva = (
            requested_real_uva and hardware_allows_real_uva and triton_launcher_allows_mapped_host_pointers()
        )
        if launcher_allows_real_uva:
            import vllm_ascend.vllm_ascend_C  # type: ignore[import-untyped]  # noqa: F401

        view = torch.ops._C_ascend.get_npu_view_from_cpu_tensor(self._cpu) if launcher_allows_real_uva else None
        self._use_real_uva = view is not None
        if launcher_allows_real_uva and not self._use_real_uva:
            logger.warning_once("Pinned CPU memory is not mapped for NPU UVA; using the async H2D fallback")
        self._uva: torch.Tensor = view if view is not None else _zeros_on_npu(self._cpu)

    def _mark_cpu_modified(self, key: int):
        self._modified_indices.add(key)

    @property
    def uses_real_uva(self) -> bool:
        """Return whether this buffer exposes mapped host storage directly."""
        return self._use_real_uva

    @property
    def cpu(self):
        return self._cpu if self._use_real_uva else MonitoredTorchTensor(self._cpu, self._mark_cpu_modified)

    @property
    def np(self):
        return self._np if self._use_real_uva else MonitoredNumPyArray(self._np, self._mark_cpu_modified)

    def _sync_dirty_rows(self) -> None:
        if not self._use_real_uva and self._modified_indices:
            dirty_rows = sorted(self._modified_indices)
            n_dirty = len(dirty_rows)
            if dirty_rows[0] == 0 and dirty_rows[-1] == n_dirty - 1:
                # Common path: dirty rows are a contiguous prefix [0..n-1].
                # This is always the case when copy_to_uva writes via
                # dst[:n] = x.  Contiguous slice copy_ keeps the CPU source
                # pinned and enables true async DMA without an intermediate
                # tensor or stream sync.
                _int64_storage(self._uva[:n_dirty]).copy_(_int64_storage(self._cpu[:n_dirty]), non_blocking=True)
            else:
                # Sparse modification pattern — fall back to indexed copy.
                # Explicitly re-pin the CPU source so that non_blocking is
                # not silently degraded. uint64 is copied through its int64
                # storage because NPU rejects DT_UINT64 in the copy kernel.
                src = self._cpu[dirty_rows].pin_memory()
                if src.dtype == torch.uint64:
                    copied = _int64_storage(src).to(device="npu", non_blocking=True)
                    _int64_storage(self._uva)[dirty_rows] = copied
                else:
                    self._uva[dirty_rows] = src.to(device="npu", non_blocking=True)
            self._modified_indices.clear()

    def uva(self, n: int | None = None) -> torch.Tensor:
        """Get the device data of the buffer.

        vLLM main (#56908) turned UvaBuffer.uva into a method taking an
        optional row count; the pool and StagedWriteTensor call it.
        """
        self._sync_dirty_rows()
        return self._uva if n is None else self._uva[:n]


vllm.v1.worker.gpu.buffer_utils.UvaBuffer = UvaBufferWrapper
