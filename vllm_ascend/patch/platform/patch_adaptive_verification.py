# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vllm-ascend project
"""Disable synthetic prompt-tail drafts for context-parallel AV admission."""

from functools import wraps

from vllm.v1.core.sched.scheduler import Scheduler


def configure_adaptive_scheduler(scheduler: Scheduler) -> None:
    config = scheduler.vllm_config
    spec = config.speculative_config
    parallel = config.parallel_config
    if (
        spec is None
        or not spec.enable_adaptive_verification
        or max(parallel.prefill_context_parallel_size, parallel.decode_context_parallel_size) <= 1
        or scheduler.dynamic_sd_lookup is not None
    ):
        return

    # Upstream's explicit per-batch draft policy already disables synthetic
    # uniform-width padding when admitting waiting requests. A constant table
    # keeps the configured draft limit for every batch size; real drafts,
    # lookahead reservations and the worker's AV budget selection are unchanged.
    # Remove this adapter when upstream exposes an AV-aware admission policy.
    scheduler.dynamic_sd_lookup = [scheduler.num_spec_tokens] * (scheduler.scheduler_config.max_num_seqs + 1)


def _install_scheduler_patch() -> None:
    original_init = Scheduler.__init__
    if getattr(original_init, "_ascend_cp_adaptive_admission", False):
        return

    @wraps(original_init)
    def initialize(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        configure_adaptive_scheduler(self)

    initialize._ascend_cp_adaptive_admission = True
    Scheduler.__init__ = initialize


_install_scheduler_patch()
