# Context Parallel Guide

## Overview

Context Parallel (CP) serves long-context requests by splitting work or KV-cache storage along the sequence dimension:

- Prefill Context Parallel (PCP) splits the prefill tokens of a long prefill request across additional ranks. Each rank computes a different part of the sequence, reducing time to first token (TTFT).
- Decode Context Parallel (DCP) shards the KV cache across ranks in a DCP group, which may reuse ranks from the PCP group, the Tensor Parallel (TP) group, or both, depending on the parallel configuration. It reduces duplicated KV-cache storage and can increase decode throughput.

For a general introduction to these two strategies, see the upstream [vLLM Context Parallel Deployment](https://docs.vllm.ai/en/latest/serving/context_parallel_deployment/) guide.

!!! warning "GQA DCP Support Removed"

    Decode Context Parallelism (DCP) for GQA models has been removed. GQA draft models maintain native TP sharding under target DCP. DCP for MLA and SFA models remains fully supported. See [RFC #21305](https://github.com/vllm-project/vllm-ascend/issues/21305).

## Supported Scenarios

### Prefill Context Parallel { #prefill-context-parallel }

PCP support is experimental. The following table shows the basic backend support and whether each feature can be combined with PCP:

| Attention Backend | Basic PCP | Prefix Caching + PCP | Chunked Prefill + PCP | MLAPO + PCP | Graph Mode + PCP | Speculative Decoding + PCP | P/D Disaggregation + PCP | KV Cache Pool + PCP | Sequence Parallelism (SP) + PCP |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MLA | ✅ Full compatibility | ✅ Full compatibility | ✅ Full compatibility | ✅ Full compatibility | 🟠 Partial compatibility (FULL not supported) | 🟠 Partial compatibility (MTP) | ✅ Full compatibility (`MooncakeConnectorV1`, `MooncakeConnectorV2`) | 🟠 Partial compatibility (`AscendStoreConnector`, non-layerwise) | ✅ Full compatibility |
| GQA | ✅ Full compatibility | ✅ Full compatibility | ✅ Full compatibility | — Not applicable | 🟠 Partial compatibility (FULL not supported) | 🟠 Partial compatibility (Eagle3) | ✅ Full compatibility (`MooncakeConnectorV1`, `MooncakeConnectorV2`) | 🟠 Partial compatibility (`AscendStoreConnector`, non-layerwise) | ✅ Full compatibility |
| SFA | ✅ Full compatibility | ✅ Full compatibility | ✅ Full compatibility | ✅ Full compatibility | 🟠 Partial compatibility (FULL not supported) | 🟠 Partial compatibility (MTP and DSpark) | ✅ Full compatibility (`MooncakeConnectorV1`, `MooncakeConnectorV2`) | 🟠 Partial compatibility (`AscendStoreConnector`, non-layerwise) | ✅ Full compatibility |
| DSA | ✅ Full compatibility | ✅ Full compatibility | ✅ Full compatibility | — Not applicable | 🟠 Partial compatibility (FULL not supported) | 🟠 Partial compatibility (MTP and DSpark) | ✅ Full compatibility (`MooncakeHybridConnector`, `MooncakeConnectorV2`) | 🟠 Partial compatibility (`AscendStoreConnector`, non-layerwise) | ✅ Full compatibility |
| DSA V4.1 | ✅ Full compatibility | ✅ Full compatibility | ✅ Full compatibility | — Not applicable | 🟠 Partial compatibility (FULL not supported) | 🟠 Partial compatibility (DSpark) | ✅ Full compatibility (`MooncakeHybridConnector`, `MooncakeConnectorV2`) | 🟠 Partial compatibility (`AscendStoreConnector`, non-layerwise) | ✅ Full compatibility |
| KDA + SFA/KPool (GLM-5.3-Flash) | ✅ Full compatibility | ✅ Full compatibility | ✅ Full compatibility | Unverified | 🟠 Partial compatibility (`FULL_DECODE_ONLY`) | 🟠 Partial compatibility (MTP) | ✅ Full compatibility (`MooncakeConnectorV2`) | 🟠 Partial compatibility (`AscendStoreConnector`, non-layerwise) | ✅ Full compatibility |
| KDA + SFA (Kimi-K3.1-Pro) | ✅ Full compatibility | ✅ Full compatibility | ✅ Full compatibility | Unverified | 🟠 Partial compatibility (`FULL_DECODE_ONLY`) | 🟠 Partial compatibility (DSpark) | ✅ Full compatibility (`MooncakeConnectorV2`) | 🟠 Partial compatibility (`AscendStoreConnector`, non-layerwise) | ✅ Full compatibility |

- ✅ **Full compatibility**: The basic path or feature combination is supported.
- 🟠 **Partial compatibility**: The basic path or feature combination is supported with the stated limitations.
- **Not applicable**: The feature does not apply to the attention backend.

### Decode Context Parallel

DCP supports eager and graph execution, prefix caching, chunked prefill, speculative decoding, P/D disaggregation, and stacking with PCP on the model and hardware combinations documented by vLLM Ascend. The following table shows whether each feature can be combined with DCP across devices and attention backends:

| Device | Attention Backend | Chunked Prefill + DCP | Prefix Caching + DCP | Graph Mode + DCP | P/D Disaggregation + DCP | Speculative Decoding + DCP | PCP + DCP |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Ascend A2/A3 | MLA | ✅ Full compatibility | ✅ Full compatibility | ✅ Full compatibility | ✅ Full compatibility | ✅ Full compatibility | ✅ Full compatibility |
| Ascend A2/A3 | GQA | ❌ Not supported | ❌ Not supported | ❌ Not supported | ❌ Not supported | ❌ Not supported | ❌ Not supported |
| Ascend A2/A3 | SFA | ✅ Full compatibility | ✅ Full compatibility | ✅ Full compatibility | ✅ Full compatibility | ✅ Full compatibility | ✅ Full compatibility |
| 950PR&950DT Products | MLA | ✅ Full compatibility | ✅ Full compatibility | ✅ Full compatibility | ✅ Full compatibility | ✅ Full compatibility | ✅ Full compatibility |
| 950PR&950DT Products | GQA | ❌ Not supported | ❌ Not supported | ❌ Not supported | ❌ Not supported | ❌ Not supported | ❌ Not supported |
| 950PR&950DT Products | SFA | ✅ Full compatibility | ✅ Full compatibility | ✅ Full compatibility | ✅ Full compatibility | ✅ Full compatibility | ✅ Full compatibility |

- ✅ **Full compatibility**: Combining the feature with DCP is supported.
- 🟠 **Partial compatibility**: Combining the feature with DCP is supported with the stated limitations.

## Usage { #usage }

### Prefill Context Parallel

Set `prefill_context_parallel_size` to the number of PCP ranks:

```bash
vllm serve <supported-model> \
    --tensor-parallel-size <tp-size> \
    --prefill-context-parallel-size <pcp-size> \
    --enforce-eager
```

Unlike DCP, PCP adds extra ranks: `world_size_with_pcp = prefill_context_parallel_size * original_world_size`.

For supported models, decode requests are sharded when
`--prefill-context-parallel-size` is greater than 1 and
`--decode-context-parallel-size` is 1. Each decode request in a batch is
assigned to one PCP rank in round-robin order, including during speculative
target verification with fixed draft lengths. When adaptive verification is
enabled, or `--decode-context-parallel-size` is greater than 1, decode requests
remain replicated across PCP ranks.

When PCP size is greater than 1, PCP stores embedding and LM Head weights as TP×PCP resident shards by default for every supported attention backend. Set `enable_pcp_embedding_lmhead_weight_sharding` to `false` to disable this behavior.

With PCP enabled, O-projection weight sharding is on by default for SFA-PCP O-projection and DSA-PCP two-stage O-projection (`wo_a`/`wo_b`). If the weight method or PCP partition does not support sharding, the option is disabled with a warning. Set `enable_pcp_o_proj_weight_sharding` to `false` to disable it. Other attention backends do not use this option. O-projection sharding is incompatible with fine-grained TP for the same modules.

#### Speculative Decoding

PCP supports MTP with MLA, SFA, and DeepSeek-V4 DSA models, Eagle3 with
GQA models, and DSpark with SFA, DeepSeek-V4 DSA, and DeepSeek-V4.1 DSA models. The target
model runs with the configured PCP topology, while the draft model is replicated on every PCP rank
and runs with a logical PCP size of `1`. Configure PCP only for the target
model. DFlash and DFlash2 also use a replicated draft with logical PCP size `1`.
DFlash2 candidate selection merges the target LM head's vocabulary shards across
both PCP and TP before computing the global candidate TopK.
For replicated DFlash/DFlash2 with `TP=1` and `DP=1`, keep EPLB disabled.

For general speculative decoding configuration and model requirements, see [Speculative Decoding](speculative_decoding.md).

##### MTP with MLA

```bash
vllm serve <mtp-capable-mla-model> \
    --tensor-parallel-size 2 \
    --prefill-context-parallel-size 2 \
    --enable-chunked-prefill \
    --enforce-eager \
    --speculative-config '{"method": "mtp", "num_speculative_tokens": 3}'
```

##### Eagle3 with GQA

```bash
vllm serve <gqa-target-model> \
    --tensor-parallel-size 2 \
    --prefill-context-parallel-size 2 \
    --enable-chunked-prefill \
    --enforce-eager \
    --speculative-config '{"method": "eagle3", "model": "<eagle3-draft-model>", "num_speculative_tokens": 3}'
```

For either method, remove `--enforce-eager` and add the following option to use a supported graph mode such as `PIECEWISE` or `FULL_DECODE_ONLY`:

```bash
--compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}'
```

##### Adaptive verification with PCP

Model Runner V2 can retain a different number of draft tokens per verification
request. Set `enable_adaptive_verification` in the speculative configuration of
a supported SFA or DeepSeek-V4 DSA model, and use `FULL_DECODE_ONLY` graphs:

```bash
--speculative-config '{"method": "mtp", "num_speculative_tokens": 5, "enable_adaptive_verification": true}' \
--compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}'
```

All compatibility changes are contained in vLLM Ascend and use unmodified
vLLM main. The Ascend platform adapter uses a constant per-batch draft limit
to disable synthetic prompt-tail admission while retaining real drafts. The
worker adapter extends the AV module's existing TP agreement to PCP without
changing its factory interface. NPU integration and performance validation of
this main-based port are still pending.

Every PCP rank receives the entire compacted verification prefix; ordinary
prefill tokens are still partitioned. For example, if requests A and B retain
3 and 1 query tokens, both ranks receive `[A0, A1, A2, B0]`. A following long
prefill request is split across the two ranks. Device query offsets `[0, 3, 4]`
describe the actual attention boundaries. CPU offsets may instead be `[0, 2, 4]`
as placeholders with the same total, while CPU scheduling lengths retain their
original upper bounds for capacity planning. The CPU placeholders must never
replace the device attention boundaries.

TP and PCP ranks agree on the cost curves, total draft budget and per-request
admitted draft counts. Graph request capacity and uniform query width come from
the selected graph descriptor, including steps that retain zero drafts. AV does
not enable an unsupported attention backend, speculative method, or topology;
the existing capability checks still apply. In particular, DeepSeek-V4.1 CP+AV,
pipeline parallelism with AV, and general dynamic draft lengths remain outside
this implementation's scope.

The four-card regression is
`tests/e2e/pull_request/four_card/context_parallel/test_accuracy_v2.py::test_sfa_pcp_adaptive_verification`.
It compares AV OFF/ON outputs with PCP2 and DCP1/DCP4, checks that all four ranks
actually trim drafts, and repeats generation to exercise prefix caching. Its
test-only cost curves deliberately favor zero drafts; they are not performance
tuning recommendations.

#### Constraints

- In P/D disaggregation, enable PCP only on the prefill (`kv_producer`) engine; the decode (`kv_consumer`) engine must use `prefill_context_parallel_size=1`.
- With PCP, `MooncakeConnectorV1` and `MooncakeHybridConnector` do not support PP. `MooncakeHybridConnector` does not support DCP.
- For PP+PCP, use `MooncakeConnectorV2` on [supported models](pipeline_parallel.md).
- KV cache pooling with PCP supports only `AscendStoreConnector` with `use_layerwise=false`.
- PCP speculative decoding supports MTP with MLA, SFA, and DeepSeek-V4 DSA models,
  Eagle3 with GQA models, DSpark with SFA, DeepSeek-V4 DSA, and DeepSeek-V4.1 DSA models,
  and replicated DFlash/DFlash2 drafts.
- PCP does not support `FULL` CUDA graphs; use eager mode, `PIECEWISE`, `FULL_AND_PIECEWISE`, or `FULL_DECODE_ONLY`.
- Encoder-decoder models and LoRA are not supported with PCP.
- PCP and DCP can be stacked for MLA and SFA. GQA does not support DCP. See
  [Combining PCP and DCP](#combining-pcp-and-dcp) for the supported sizes.
- Adaptive verification with PCP requires the backend/graph support and Ascend
  compatibility adapters described above.
- Dynamic draft lengths are outside the currently validated scope.

### Decode Context Parallel

#### Standalone DCP (PCP=1)

Use the following topology when PCP is disabled (`prefill_context_parallel_size=1`):

```bash
vllm serve <glm-5.2-model> \
  --tensor-parallel-size <N> \
  --prefill-context-parallel-size 1 \
  --decode-context-parallel-size <N> \
  --block-size <B> \
  --cp-kv-cache-interleave-size <B>
```

Standalone DCP reuses only the TP ranks and does not increase the world size.

- For an MLA model:
    - `tensor_parallel_size >= decode_context_parallel_size`
    - `tensor_parallel_size % decode_context_parallel_size == 0`
- For a sparse SFA model with DCP enabled (`decode_context_parallel_size>1`):
    - `decode_context_parallel_size == tensor_parallel_size`
- For GQA/MQA models:
    - Decode Context Parallelism (DCP) is not supported. GQA/MQA models must keep `decode_context_parallel_size=1`.

#### Combining PCP and DCP { #combining-pcp-and-dcp }

When both PCP and DCP are enabled (`prefill_context_parallel_size>1` and `decode_context_parallel_size>1`), their supported sizes depend on the attention backend. The standalone DCP constraints above do not apply to this topology. Here, `PCP_SIZE` is `prefill_context_parallel_size`, `DCP_SIZE` is `decode_context_parallel_size`, and `TP_SIZE` is `tensor_parallel_size`:

- **MLA and SFA**: Support `DCP_SIZE = PCP_SIZE` and `DCP_SIZE = PCP_SIZE * TP_SIZE`.
- **GQA**: Decode Context Parallelism (DCP) is not supported for GQA models.

DCP reuses the ranks introduced by PCP and TP, so stacking DCP does not add ranks beyond the PCP world size. For example, `TP2 + PCP2 + DCP4` is a supported MLA/SFA topology.

#### KV-cache Transfer

The following constraint applies to both standalone DCP and PCP+DCP:

- In a KV-cache transfer scenario such as KV pooling or P/D disaggregation, set `cp_kv_cache_interleave_size` to the KV-cache `block_size` (default: 128):

    ```shell
    vllm serve deepseek-ai/DeepSeek-V2-Lite \
        --tensor-parallel-size 2 \
        --decode-context-parallel-size 2 \
        --cp-kv-cache-interleave-size 128 \
        --kv-transfer-config '{...}'
    ```

- Mooncake block-key layerwise KV pooling supports standalone DCP when the producer and consumer use the same DCP size. PCP and DCP cannot be enabled together for layerwise pooling, and cache groups whose block size is not scaled by DCP (sliding-window, recurrent/align-state, circular-buffer, ...) are not supported with DCP. The smallest cache-group block size must divide every other group block size, and `--prefix-match-unit` must be unset with DCP. GQA/MQA models do not support DCP. Hybrid DCP support is experimental. See [Mooncake Layerwise KV Cache Pooling](mooncake_layerwise_adaptation_and_optimization.md#decode-context-parallelism-dcp).

For implementation details, see the [Context Parallel design document](../../developer_guide/Design_Documents/context_parallel.md).
