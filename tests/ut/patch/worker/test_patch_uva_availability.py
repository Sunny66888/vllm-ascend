# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib.metadata

from vllm_ascend.patch.worker.patch_v2.patch_uva import is_uva_available


def test_uva_available_when_registered_pinned_memory_is_requested(monkeypatch):
    monkeypatch.setenv(
        "PYTORCH_NPU_ALLOC_CONF",
        "expandable_segments:True,pinned_mem_register:True",
    )

    assert is_uva_available()


def test_uva_unavailable_without_registered_pinned_memory(monkeypatch):
    monkeypatch.setenv("PYTORCH_NPU_ALLOC_CONF", "expandable_segments:True")

    assert not is_uva_available()


def test_uva_availability_does_not_query_triton_version(monkeypatch):
    monkeypatch.setenv("PYTORCH_NPU_ALLOC_CONF", "pinned_mem_register:True")

    def fail_on_version_query(_):
        raise AssertionError("unexpected package-version query")

    monkeypatch.setattr(importlib.metadata, "version", fail_on_version_query)

    # The mapped view is NPU-typed before reaching Triton. Availability must
    # therefore depend on runtime mapping, not the installed Triton version.
    assert is_uva_available()
