# ModelRunner V2 UVA 分支状态与验证说明

本文记录 `modelrunner-v2-uva` 分支当前的代码状态、服务器容器环境、启动失败原因，以及后续验证该分支的方法。本文面向开发和代码评审，不包含服务器凭据。

## 1. 当前结论

当前分支代码已经同步到服务器，容器、模型文件和 NPU 资源基本可用。但是，当前容器中的 vLLM 版本与分支代码预期的 upstream vLLM 版本不一致，因此服务在启动阶段发生 Python 导入错误，尚未进入模型加载和推理验证阶段。

当前结论如下：

| 项目 | 当前状态 |
| --- | --- |
| 分支 | `modelrunner-v2-uva` |
| 当前提交 | `31b1b4686236ba729dfd7ec58be337c899e1ec3e` |
| 工作区 | 干净，无未提交修改 |
| 服务器代码目录 | `/data/gry/vllm-ascend` |
| 容器 | `test-g`，运行中 |
| 模型 | `/mnt/weight/Qwen3-30B-A3B`，文件存在 |
| NPU | 16 张 Ascend 910，设备状态正常 |
| 容器内 vLLM | `0.30.0` |
| 容器内 vLLM-Ascend | `0.30.0rc2.dev10+g6d4339aa1.d20261012` |
| 容器内 Triton-Ascend | `3.6.0+dev20261007182438` |
| Triton UVA 判断 | 构建时间晚于修复确认时间，应已移除 `location.type` 校验；仍需源码检查和 NPU 测试确认 |
| 分支兼容基线 | 当前 main 线对应的 vLLM `v0.31.0` 兼容逻辑 |
| 当前代码加载方式 | 使用 `PYTHONPATH` 加载 `/data/gry/vllm-ascend` |
| 正式推理验证 | 尚未完成 |

当前不需要通过 `pip install -e` 才能加载分支代码。验证时可以使用：

```bash
export PYTHONPATH=/data/gry/vllm-ascend:$PYTHONPATH
```

这样可以让 Python 优先加载工作区中的 vLLM Ascend 源码。

## 2. 服务启动失败原因

启动日志中的关键错误是：

```text
ImportError: cannot import name 'engram_head_shard_rank'
from vllm.models.deepseek_v41.common.engram
```

容器中的 vLLM `0.30.0` 将该符号放在：

```text
vllm.models.deepseek_v41.nvidia.engram
```

而当前分支的兼容逻辑在相应版本判断下尝试从：

```text
vllm.models.deepseek_v41.common.engram
```

导入，因此在服务初始化阶段失败。

这表明当前问题是 upstream vLLM API/模块布局不匹配：

```text
当前分支代码：预期 vLLM v0.31.0 或相应 verified commit
当前容器环境：实际安装 vLLM v0.30.0
```

该问题不是 NPU 数量、显存、HCCL、端口、`bfloat16` 参数或 UVA 映射本身导致的，也不是代理配置导致的 editable 安装问题。当前分支中这段 DeepSeek 兼容逻辑来自更早的 main 同步提交，而不是本分支的 UVA 提交。

## 3. `modelrunner-v2-uva` 分支的修改

该分支相对本地 `main` 只有一个核心提交：

```text
31b1b4686236ba729dfd7ec58be337c899e1ec3e
feat: enable ModelRunner v2 UVA on supported stacks
```

修改目标是：在支持 pinned memory 注册和 CPU-NPU 映射的 Ascend 环境中，为 ModelRunner V2 启用真实 UVA；如果当前运行环境无法建立映射，则保留异步 H2D 拷贝回退路径。

### 3.1 UVA 可用性从版本判断改为运行时判断

核心文件：[`patch_uva.py`](../../../../vllm_ascend/patch/worker/patch_v2/patch_uva.py)

旧逻辑包括：

- 检查 `PYTORCH_NPU_ALLOC_CONF` 是否启用 `pinned_mem_register:True`；
- 根据 `triton-ascend` 版本号直接禁用某些版本的 UVA；
- 用软件包版本推断当前环境是否可以使用 UVA。

新逻辑包括：

- 仍然要求用户启用 pinned memory 注册；
- 不再根据 Triton 版本号硬编码禁用 UVA；
- 使用 `torch.ops._C_ascend.get_npu_view_from_cpu_tensor(...)` 尝试建立 NPU view；
- 映射成功时使用真实 UVA；
- 映射失败时自动退回异步 H2D 拷贝，并发出一次警告。

判断方式因此从“该软件版本理论上是否支持”改为“当前运行环境实际是否能建立映射”。

### 3.2 `UvaBufferWrapper` 的运行时行为

`UvaBufferWrapper` 主要完成以下工作：

- 创建 pinned CPU tensor；
- 尝试获取对应的 NPU view；
- 使用 NPU view 访问 CPU pinned memory；
- 对发生修改的行执行同步；
- 连续行使用非阻塞拷贝；
- 稀疏行使用索引拷贝，并重新 pin CPU 源数据；
- 对 `torch.uint64` 在必要时转换为适合 NPU kernel 的视图；
- 通过 `uva(n=None)` 返回完整 buffer 或前 `n` 行。

最后，分支将 upstream vLLM 的：

```python
vllm.v1.worker.gpu.buffer_utils.UvaBuffer
```

替换为 Ascend 侧的 `UvaBufferWrapper`。

因此，虽然分支名称是 `modelrunner-v2-uva`，但本次没有直接重写 `vllm_ascend/worker/v2/model_runner.py` 的主体逻辑，而是通过 patch 修改 ModelRunner V2 使用的底层 UVA buffer 行为。

### 3.3 测试修改

文件：[`test_uva.py`](../../../../tests/e2e/pull_request/one_card/model_runner_v2/test_uva.py)

主要变化：

- 新增 `check_real_uva_mapping()` worker 检查；
- 在 worker 的 NPU 上下文中实际创建 `UvaBufferWrapper`；
- 验证真实 UVA 映射是否建立，以及 buffer 是否位于 NPU 设备；
- 同时覆盖 eager mode 和 graph mode；
- 设置 `PYTORCH_NPU_ALLOC_CONF=pinned_mem_register:True`；
- 设置 `VLLM_USE_V2_MODEL_RUNNER=1`；
- 通过 `worker_extension_cls` 调用 worker 侧检查；
- 保留 Qwen3-0.6B 的生成测试。

文件：[`test_uva_view.py`](../../../../tests/e2e/nightly/single_node/ops/singlecard_ops/test_uva_view.py)

该测试删除了对 Triton Ascend `3.2.1` 和 `3.2.2` 的硬编码跳过逻辑，改为依赖实际运行时映射结果。

文件：[`test_patch_uva_availability.py`](../../../../tests/ut/patch/worker/test_patch_uva_availability.py)

新增单元测试覆盖：

- 启用 `pinned_mem_register:True` 时 UVA 可用；
- 没有该配置时 UVA 不可用；
- UVA 可用性判断不再依赖 Triton 包版本查询。

### 3.4 CI 配置修改

分支还做了两项 CI 调整：

- 从跳过列表中移除 ModelRunner V2 UVA E2E 测试；
- 将该测试的预计耗时从 80 秒调整为 160 秒。

## 4. `triton-ascend` 版本对 UVA 的影响

ModelRunner V2 的真实 UVA 路径会把注册后的 pinned CPU memory 映射成 NPU view，随后将该 view 作为参数传给 Triton kernel。部分 `triton-ascend` 版本会在启动 kernel 前检查指针对应的 `location.type`，并要求所有输入指针都属于 NPU device memory。这个检查会把 UVA 场景中的 pinned host memory 错误地拒绝掉。

该问题只影响 Triton kernel 是否接受映射后的 host pointer，不代表 C++ 层的 `aclrtHostGetDevicePointer` 映射一定失败。也就是说，可能出现以下情况：

```text
CPU pinned memory 注册成功
  -> NPU view 创建成功
  -> Triton launcher 因 location.type 校验拒绝该指针
```

### 4.1 版本兼容性

| `triton-ascend` 版本或构建 | UVA 影响 | 建议 |
| --- | --- | --- |
| `< 3.2.1` | 尚未引入该 `location.type` 校验，因此不受这个特定问题影响 | 版本较旧，不建议仅为 UVA 降级 |
| `3.2.1` | 存在错误的 device-only pointer 校验，会拒绝 UVA pinned host pointer | 不用于真实 UVA |
| 最初发布的官方 `3.2.2` wheel | 发布时仍包含该校验 | 不用于真实 UVA |
| 2026 年 9 月及之后、基于最新 `release/3.2.2` 构建的日构建 | 已移除该校验，但有些内部 wheel 可能仍把元数据版本写成 `3.2.2` | 必须通过来源、哈希或安装内容确认 |
| 官方 `3.2.2.post2` | 包含后续修复，可与原始 `3.2.2` 明确区分 | `3.2.x` 线上推荐版本 |
| 2026 年 9 月及之后的 `3.6.0` 日构建 | 已移除该校验 | 可以用于 UVA，但仍需验证完整软件栈 |
| 官方 `3.6.0` | 已包含修复 | 新版本软件栈可优先使用 |

需要特别注意 Python 包版本的精确匹配规则：

```text
triton-ascend==3.2.2
```

只会选择版本元数据严格匹配 `3.2.2` 的包，不会自动选择：

```text
triton-ascend==3.2.2.post2
```

因此，“`release/3.2.2` 分支上的最新源码已经修复”和“安装 `triton-ascend==3.2.2` 一定得到修复后的 wheel”是两个不同结论。对于公共依赖和可复现构建，应显式使用 `3.2.2.post2`；如果使用版本号仍为 `3.2.2` 的内部重构 wheel，应记录下载地址和 SHA256。

### 4.2 当前服务器镜像中的版本判断

当前镜像返回的信息是：

```text
metadata version: 3.6.0+dev20261007182438
dist-info: /usr/local/python3.11.10/lib/python3.11/site-packages/
           triton_ascend-3.6.0+dev20261007182438.dist-info
wheel: triton_ascend-3.6.0+dev20261007182438-cp311-cp311-
       manylinux_2_27_aarch64.manylinux_2_28_aarch64.whl
sha256: 39dbee1d3cef09a9a36dbaef85ee6d5095763b5edca2151235a1e78c08d3157c
```

该版本是 2026 年 10 月 7 日的 `3.6.0` 日构建，晚于 Triton-Ascend 维护者在 2026 年 9 月 18 日确认 `3.6` 日构建已移除 `location.type` 校验的时间。因此，从版本和构建时间判断，该 wheel 应当已经包含 UVA 修复，并且不会被当前分支按版本号禁用。

导入 `triton` 时出现的以下信息：

```text
FutureWarning: triton.language.extra.cann.extension.* is deprecated
```

只是 CANN 扩展 API 的弃用提示，与 UVA pointer 校验无关。

由于该 wheel 的 `direct_url.json` 只记录了容器内的 `file:///tmp/...` 临时路径，并未记录上游 Git commit，SHA256 也没有公开的 commit 对照表，所以仍建议在镜像内检查实际安装内容：

```bash
TRITON_ROOT=$(PYTHONWARNINGS=ignore::FutureWarning python -c \
'import pathlib,triton; print(pathlib.Path(triton.__file__).resolve().parent)')

grep -RnsC 8 \
  --include='*.py' \
  -E 'location\.type|aclrtPointerGetAttributes' \
  "$TRITON_ROOT/backends/ascend"
```

判断标准：

- 如果仍存在“输入指针的 `location.type` 必须是 device memory，否则抛出异常”的逻辑，则该构建不能用于真实 UVA；
- 如果没有这段参数校验，则说明 Triton 侧修复已经包含；
- 如果关键字出现在其他内存管理路径中，需要结合前后代码判断，不能仅凭关键字存在就认定 UVA 不可用。

### 4.3 Triton 修复不等于完整 UVA 链路通过

即使 `triton-ascend` 已移除校验，真实 UVA 仍要求：

- `PYTORCH_NPU_ALLOC_CONF` 包含 `pinned_mem_register:True`；
- `torch_npu` 包含 pinned host registration 和映射相关修复；
- CANN 版本不低于 9.0；
- Driver 版本不低于 26.0，且满足 UVA 对商用驱动版本的要求；
- vLLM Ascend C++ 扩展能够通过 `aclrtHostGetDevicePointer` 创建 NPU view；
- Triton kernel 能直接读取该 view，并观察到 CPU 对 pinned memory 的更新。

在 NPU 容器中可以使用以下测试验证最后两项：

```bash
export PYTORCH_NPU_ALLOC_CONF=pinned_mem_register:True

pytest -sv \
  tests/e2e/nightly/single_node/ops/singlecard_ops/test_uva_view.py \
  -k "npu_view_reads_cpu_updates_without_copy or npu_view_keeps_cpu_storage_alive"
```

只有这些测试实际执行并通过，才能确认当前镜像中的 Triton、TorchNPU、CANN、Driver 和 vLLM Ascend 映射实现共同打通了真实 UVA。仅看到版本号符合预期，不应直接作为完整 UVA 验证通过的证据。

## 5. 与 vLLM 0.30.0 release 线的关系

当前容器更接近以下组合：

```text
vLLM v0.30.0
vLLM-Ascend v0.30.0rc2 系列
```

而当前 `modelrunner-v2-uva` 分支来自更新的 main 代码线，仓库元数据已经按 vLLM `v0.31.0` 兼容逻辑组织。因此，不能直接把当前 main 衍生分支原样放入 vLLM `0.30.0` 容器中并期待所有模块兼容。

正确的验证路线有两种：

### 路线 A：验证当前 `modelrunner-v2-uva` 分支

使用与当前分支匹配的 vLLM `v0.31.0` 或对应 verified commit，然后通过 `PYTHONPATH` 加载工作区代码：

```bash
export PYTHONPATH=/data/gry/vllm-ascend:$PYTHONPATH
```

这验证的是：

```text
当前 modelrunner-v2-uva 分支 + 正确的 vLLM upstream 基线
```

### 路线 B：继续使用当前 vLLM 0.30.0 容器

使用 vLLM-Ascend `releases/v0.30.0rc2` 等对应的 release 基线，再将 UVA 修改移植到该基线。

这验证的是：

```text
vLLM v0.30.0 + vLLM-Ascend v0.30.0rc2 + UVA 修改
```

不能把路线 A 的分支代码直接当作路线 B 的代码使用，除非额外补充 upstream vLLM 0.30.0 的兼容处理。

## 6. 后续验证顺序

在正式运行 Qwen3-30B-A3B 服务前，建议按以下顺序进行：

1. 确认容器中的 upstream vLLM 版本与分支兼容基线一致。
2. 设置 `PYTHONPATH`，确认 `import vllm_ascend` 实际来自 `/data/gry/vllm-ascend`。
3. 运行 UVA 可用性单元测试。
4. 运行 ModelRunner V2 UVA 的单卡 E2E 测试，覆盖 eager 和 graph mode。
5. 确认 `PYTORCH_NPU_ALLOC_CONF=pinned_mem_register:True` 已设置。
6. 再启动 Qwen3-30B-A3B 服务。
7. 启动成功后，执行健康检查和最小推理请求，再进行长时间或性能测试。

在版本基线未统一前，继续调整服务启动参数没有意义，因为当前错误发生在模型加载前的 Python import 阶段。

## 7. 当前验证边界

截至本文记录时：

- 代码同步已确认；
- 容器、模型文件和 NPU 资源已确认存在；
- 服务曾尝试启动，但在 DeepSeek 相关 import 阶段失败；
- 没有成功完成模型加载；
- 没有成功执行 Qwen3-30B-A3B 推理请求；
- 当前没有可以宣称“UVA 验证通过”的结果。

因此，后续报告中应将当前状态描述为“代码和环境已准备，验证被 upstream vLLM 版本不匹配阻塞”，而不是“UVA 功能已验证通过”。

## 参考资料

- [vLLM-Ascend v0.30.0rc1 release checklist](https://github.com/vllm-project/vllm-ascend/issues/17749)
- [vLLM-Ascend v0.30.0rc2 environment update](https://github.com/vllm-project/vllm-ascend/issues/18977)
- [Triton-Ascend UVA pointer validation issue #783](https://github.com/triton-lang/triton-ascend/issues/783)
- [Triton-Ascend PR #1759](https://github.com/triton-lang/triton-ascend/pull/1759)
- [Triton-Ascend official releases](https://github.com/triton-lang/triton-ascend/releases)
- [Python package version specifiers](https://packaging.python.org/en/latest/specifications/version-specifiers/)
