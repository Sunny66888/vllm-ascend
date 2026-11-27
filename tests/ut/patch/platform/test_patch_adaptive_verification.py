# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vllm-ascend project

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorBase_V1
from vllm.v1.core.sched import scheduler as scheduler_module
from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.single_type_kv_cache_manager import register_all_kvcache_specs
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig, KVCacheGroupSpec
from vllm.v1.outputs import DraftTokenIds
from vllm.v1.structured_output import StructuredOutputManager

from tests.ut.kv_offload.utils import create_model_runner_output, create_request, create_vllm_config
from vllm_ascend.patch.platform import patch_adaptive_verification as admission_patch


@pytest.mark.parametrize("adaptive", [False, True])
@pytest.mark.parametrize("pcp,dcp", [(1, 1), (2, 1), (1, 2)])
@pytest.mark.parametrize("existing", [None, [3, 2, 1]])
def test_adaptive_admission_policy_preserves_draft_limit(adaptive, pcp, dcp, existing):
    config = SimpleNamespace(
        speculative_config=SimpleNamespace(enable_adaptive_verification=adaptive),
        parallel_config=SimpleNamespace(prefill_context_parallel_size=pcp, decode_context_parallel_size=dcp),
    )
    scheduler = SimpleNamespace(
        vllm_config=config,
        scheduler_config=SimpleNamespace(max_num_seqs=2),
        num_spec_tokens=3,
        num_lookahead_tokens=3,
        dynamic_sd_lookup=existing,
    )
    admission_patch.configure_adaptive_scheduler(scheduler)
    if adaptive and max(pcp, dcp) > 1 and existing is None:
        assert scheduler.dynamic_sd_lookup == [3, 3, 3]
    else:
        assert scheduler.dynamic_sd_lookup is existing
    assert scheduler.num_spec_tokens == scheduler.num_lookahead_tokens == 3


def _make_scheduler(monkeypatch, *, adaptive, async_scheduling, budget=128, matched_tokens=None):
    config = create_vllm_config(
        max_num_seqs=1 if budget == 1 else 2,
        max_num_batched_tokens=128,
        block_size=16,
        speculative_method="ngram",
        num_speculative_tokens=3,
    )
    # CPU admission coverage: set AV after CPU fixture normalization;
    # backend/graph capability validation is covered by worker tests.
    config.speculative_config.enable_adaptive_verification = adaptive
    config.parallel_config.prefill_context_parallel_size = 2
    config.scheduler_config.async_scheduling = async_scheduling
    config.scheduler_config.max_num_batched_tokens = budget
    config.cache_config.num_gpu_blocks = 128
    if matched_tokens is None:
        config.kv_transfer_config = None
    else:
        connector = MagicMock(spec=KVConnectorBase_V1)
        connector.requires_kv_delivery = False
        connector.supports_divergent_local_hybrid_hits = False
        connector.get_num_new_matched_tokens.return_value = (matched_tokens, False)
        connector.build_connector_meta.return_value = None
        connector.take_events.return_value = ()
        monkeypatch.setattr(scheduler_module.KVConnectorFactory, "create_connector", lambda **kwargs: connector)
    register_all_kvcache_specs(None)
    kv_config = KVCacheConfig(
        num_blocks=128,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["attention"], FullAttentionSpec(block_size=16, num_kv_heads=1, head_size=1, dtype=torch.float32)
            )
        ],
    )
    scheduler_cls = AsyncScheduler if async_scheduling else admission_patch.Scheduler
    return scheduler_cls(
        vllm_config=config,
        kv_cache_config=kv_config,
        block_size=16,
        structured_output_manager=StructuredOutputManager(config),
    )


@pytest.mark.parametrize("adaptive", [False, True])
@pytest.mark.parametrize("async_scheduling", [False, True])
@pytest.mark.parametrize("drafts", [[1], [1, 2, 3]])
def test_cached_prompt_tail_preserves_real_drafts(monkeypatch, adaptive, async_scheduling, drafts):
    scheduler = _make_scheduler(monkeypatch, adaptive=adaptive, async_scheduling=async_scheduling)
    first = create_request(1, num_tokens=33, max_tokens=16)
    second = create_request(1, num_tokens=33, max_tokens=16)
    second.request_id = "second"
    scheduler.add_request(first)
    prefill = scheduler.schedule()
    assert prefill.num_scheduled_tokens[first.request_id] == 33
    scheduler.update_from_output(prefill, create_model_runner_output([first]))
    scheduler.update_draft_token_ids(DraftTokenIds([first.request_id], [drafts]))
    scheduler.add_request(second)
    output = scheduler.schedule()
    assert output.scheduled_spec_decode_tokens[first.request_id] == drafts
    assert output.num_spec_tokens_to_schedule == 3
    assert output.num_scheduled_tokens[second.request_id] == (1 if adaptive else 4)
    assert output.scheduled_spec_decode_tokens.get(second.request_id, []) == ([] if adaptive else [-1] * 3)


@pytest.mark.parametrize("adaptive", [False, True])
@pytest.mark.parametrize("async_scheduling", [False, True])
@pytest.mark.parametrize("budget", [1, 128])
def test_external_cache_prompt_tail_admission(monkeypatch, adaptive, async_scheduling, budget):
    scheduler = _make_scheduler(
        monkeypatch, adaptive=adaptive, async_scheduling=async_scheduling, budget=budget, matched_tokens=32
    )
    request = create_request(1, num_tokens=33, max_tokens=16)
    scheduler.add_request(request)
    output = scheduler.schedule()
    expected = {request.request_id: 1 if adaptive else 4} if adaptive or budget > 1 else {}
    assert output.num_scheduled_tokens == expected
    assert output.num_spec_tokens_to_schedule == 3
    assert output.scheduled_spec_decode_tokens.get(request.request_id, []) == (
        [-1] * 3 if not adaptive and budget > 1 else []
    )


def test_adaptive_scheduler_constructor_patch_is_idempotent():
    initializer = admission_patch.Scheduler.__init__
    admission_patch._install_scheduler_patch()
    assert admission_patch.Scheduler.__init__ is initializer
    assert initializer._ascend_cp_adaptive_admission
