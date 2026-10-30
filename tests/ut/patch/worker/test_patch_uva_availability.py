# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

from vllm_ascend.patch.worker.patch_v2 import patch_uva


def test_real_uva_requested_when_registered_pinned_memory_is_configured(monkeypatch):
    monkeypatch.setenv(
        "PYTORCH_NPU_ALLOC_CONF",
        "expandable_segments:True,pinned_mem_register:True",
    )

    assert patch_uva.is_real_uva_requested()


def test_real_uva_not_requested_without_registered_pinned_memory(monkeypatch):
    monkeypatch.setenv("PYTORCH_NPU_ALLOC_CONF", "expandable_segments:True")

    assert not patch_uva.is_real_uva_requested()


def test_real_uva_not_requested_for_non_exact_allocator_option(monkeypatch):
    monkeypatch.setenv("PYTORCH_NPU_ALLOC_CONF", "pinned_mem_register:True-ish")

    assert not patch_uva.is_real_uva_requested()


def test_triton_launcher_rejecting_host_pointers_is_detected(monkeypatch, tmp_path):
    driver = tmp_path / "driver.py"
    driver.write_text(
        "attributes.location.type\nPointer argument cannot be accessed from Triton",
        encoding="utf-8",
    )
    monkeypatch.setattr(patch_uva, "find_spec", lambda _: SimpleNamespace(origin=str(driver)))
    patch_uva.triton_launcher_allows_mapped_host_pointers.cache_clear()

    try:
        assert not patch_uva.triton_launcher_allows_mapped_host_pointers()
    finally:
        patch_uva.triton_launcher_allows_mapped_host_pointers.cache_clear()


def test_triton_launcher_without_host_pointer_rejection_is_accepted(monkeypatch, tmp_path):
    driver = tmp_path / "driver.py"
    driver.write_text("def launch(): ...", encoding="utf-8")
    monkeypatch.setattr(patch_uva, "find_spec", lambda _: SimpleNamespace(origin=str(driver)))
    patch_uva.triton_launcher_allows_mapped_host_pointers.cache_clear()

    try:
        assert patch_uva.triton_launcher_allows_mapped_host_pointers()
    finally:
        patch_uva.triton_launcher_allows_mapped_host_pointers.cache_clear()


def test_missing_triton_launcher_keeps_copy_fallback(monkeypatch):
    monkeypatch.setattr(patch_uva, "find_spec", lambda _: None)
    patch_uva.triton_launcher_allows_mapped_host_pointers.cache_clear()

    try:
        assert not patch_uva.triton_launcher_allows_mapped_host_pointers()
    finally:
        patch_uva.triton_launcher_allows_mapped_host_pointers.cache_clear()
