# RAG^2（Recursive & Refined RAG）

单卡环境下的多跳问答流水线：云端大模型规划与循证生成，本地语义检索 + NLI 逻辑门控。

## 目录结构

```
RAG2/
├── config.py          # 全局参数
├── preprocess.py      # 数据预处理与切片
├── indexing.py        # FAISS索引构建
├── retriever.py       # 语义检索
├── filter.py          # NLI逻辑门控
├── generator.py       # LLM生成封装
├── openai_client.py   # API客户端
├── main.py            # 主入口
├── evaluator.py       # 评估脚本
├── data/
│   ├── hotpotqa.json          # 完整数据集
│   └── optimal_hotpotqa.json  # 高分样本（500条）
├── indexes/           # 索引存储
└── output/            # 评估结果
```
## MiMo-V2-Flash api限时免费

地址：https://platform.xiaomimimo.com/#/console

## 快速开始

```bash
pip install -r requirements.txt
python main.py
```

## 构建索引

```bash
python -c "from main import prepare_index; prepare_index('data/hotpotqa.json')"
```

## 评估

```bash
# 完整评估
python evaluator.py --dataset data/hotpotqa.json

# 部分评估
python evaluator.py --sample-size 50

# 指定输出
python evaluator.py --output output/results.json
```

**指标：** Contain-ACC | F1-Score | ROUGE-L

## 依赖

Python 3.9+ | torch | transformers | faiss-cpu | openai
