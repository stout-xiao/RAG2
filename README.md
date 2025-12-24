# RAG^2（Recursive & Refined RAG）

单卡环境下的多跳问答（Multi-hop QA）流水线：云端大模型负责规划与循证生成，本地语义检索 + NLI 逻辑门控负责证据筛选与漂移抑制。

## 目录结构
- `config.py`：全局参数（模型、阈值、限流、索引路径、设备）。
- `preprocess.py`：数据加载与句级滑动窗口切片，输出带 `doc_title`/`chunk_id` 的 chunk 列表。
- `indexing.py`：调用 contriever 编码，构建 FAISS `IndexHNSWFlat`，保存索引与元数据。
- `retriever.py`：语义检索与 `StateTracker` 占位符解析（`[ANSWER_i]`）。
- `filter.py`：DeBERTa MNLI 逻辑门控，自适应阈值 `max(tau_min, mean + 0.5*std)`，低分回退。
- `generator.py`：云端 LLM 封装（分解、循证生成、锚点提取），带简单限流。
- `main.py`：调度入口，`prepare_index` 预构建索引，`solve_question` 完整 RAG² 流程。
- `evaluator.py`：Contain-ACC 与 token-level F1 指标。
- `data/`：示例数据（`hotpotqa.json`、`musique.json`）。

## 依赖与环境
- Python 3.9+，`torch`，`transformers`，`faiss-cpu`（或 GPU 版）。
- 模型（可本地路径或 HuggingFace 名称）：
  - EMBED：`facebook/contriever-msmarco`
  - NLI：`microsoft/deberta-large-mnli`
- 环境变量（可选）：`GENERATOR_API_BASE`、`GENERATOR_API_KEY`、`EMBED_MODEL`、`NLI_MODEL`。

## 准备步骤
1. 安装依赖（示例）：
   ```bash
   pip install torch transformers faiss-cpu
   ```
2. 准备本地模型（离线环境请提前下载并将路径写入 `EMBED_MODEL`、`NLI_MODEL`）。
3. 配置云端 LLM 访问（若需真实分解/生成）：设置 `GENERATOR_API_BASE`、`GENERATOR_API_KEY` 或在代码中注入自定义 `call_fn`。

## 构建索引
```bash
python -c "from main import prepare_index; prepare_index('data/hotpotqa.json')"
```
默认会生成 `faiss.index` 与 `faiss_meta.json`。Musique 同理，路径替换即可。

## 运行示例
```bash
python - <<'PY'
from main import solve_question

# 如需真实 LLM，可在 Generator(call_fn=...) 传入自定义调用函数。
ans, evidence, state = solve_question("Who founded the company that created the HTTP protocol?")
print("Answer:", ans)
print("State:", state)
print("Top evidence:", [e["doc_title"] for e in evidence[:3]])
PY
```

## 逻辑要点
- 切片：Hotpot 每 4 句重叠 1 句；MuSiQue 每 3 句重叠 1 句；保留 `doc_title`/`chunk_id`。
- 检索：HNSW 近似检索，`efSearch` 可在 `config.py` 调整。
- 逻辑门控：NLI 生成 `entail_score`，自适应阈值过滤；全低分时回退最高分并标记 `low_confidence`。
- 状态递归：`StateTracker` 用前一跳锚点替换 `[ANSWER_i]` 占位符，递归检索。
- 生成：提示要求“仅基于 Provided Context”，输出推理步骤 + 最终答案。

## 评估
使用 `evaluator.evaluate(pairs)` 传入 `(pred, gold)` 可得 Contain-ACC 与 token-level F1 均值。

## 常见调整
- 资源受限：将 `config.fp16=False`、`device='cpu'`，或降低 `faiss_m`/`efSearch`。
- 召回/精度：增大检索 `k`，或调高/调低 `lambda_weight`、`tau_min`。
- 长文本：视需要提高切片大小或改用更强 encoder。
