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

## 4. 与 vLLM 0.30.0 release 线的关系

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

## 5. 后续验证顺序

在正式运行 Qwen3-30B-A3B 服务前，建议按以下顺序进行：

1. 确认容器中的 upstream vLLM 版本与分支兼容基线一致。
2. 设置 `PYTHONPATH`，确认 `import vllm_ascend` 实际来自 `/data/gry/vllm-ascend`。
3. 运行 UVA 可用性单元测试。
4. 运行 ModelRunner V2 UVA 的单卡 E2E 测试，覆盖 eager 和 graph mode。
5. 确认 `PYTORCH_NPU_ALLOC_CONF=pinned_mem_register:True` 已设置。
6. 再启动 Qwen3-30B-A3B 服务。
7. 启动成功后，执行健康检查和最小推理请求，再进行长时间或性能测试。

在版本基线未统一前，继续调整服务启动参数没有意义，因为当前错误发生在模型加载前的 Python import 阶段。

## 6. 当前验证边界

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
