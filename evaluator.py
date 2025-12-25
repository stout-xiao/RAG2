"""
RAG系统性能评估脚本
评估指标：Contain-ACC、F1-score、ROUGE-L

使用方法：
    # 评估全部数据集
    python evaluator.py --dataset data/hotpotqa.json

    # 评估指定数量的样本
    python evaluator.py --sample-size 50

    # 指定输出文件
    python evaluator.py --output output/my_results.json
"""
from __future__ import annotations

import json
import os
import re
import warnings
import logging
from typing import Dict, Iterable, Tuple

# 抑制 transformers 模型加载警告
warnings.filterwarnings("ignore", message="Some weights of the model checkpoint")
logging.getLogger("transformers").setLevel(logging.ERROR)

from main import solve_question


# =============================================================================
# 基础评估函数
# =============================================================================

def normalize(text: str) -> str:
    """文本规范化：小写、去标点、去冠词、压缩空白"""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\u4e00-\u9fff\s]", " ", text)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def contain_acc(pred: str, gold: str) -> bool:
    """
    Contain-ACC：检查标准答案是否包含在预测答案中
    
    物理含义：硬命中判断，标准答案是否作为子串出现在预测中
    """
    p = normalize(pred)
    g = normalize(gold)
    return g in p if g else False


def token_f1(pred: str, gold: str) -> float:
    """
    Token-level F1：词级别的F1分数
    
    物理含义：预测和标准答案在词集合层面的重叠程度（不考虑词序）
    计算逻辑：F1 = 2 * Precision * Recall / (Precision + Recall)
    """
    pred_tokens = normalize(pred).split()
    gold_tokens = normalize(gold).split()
    if not pred_tokens or not gold_tokens:
        return 0.0
    
    # 计算多重集合的重叠
    common = {}
    for t in pred_tokens:
        common[t] = common.get(t, 0) + 1
    overlap = 0
    for t in gold_tokens:
        if common.get(t, 0) > 0:
            overlap += 1
            common[t] -= 1
    
    precision = overlap / len(pred_tokens)
    recall = overlap / len(gold_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def rouge_l(pred: str, gold: str) -> float:
    """
    ROUGE-L：基于最长公共子序列的分数
    
    物理含义：预测和标准答案在保持词序的情况下的匹配程度
    计算逻辑：用动态规划求LCS长度，计算基于LCS的F1分数
    """
    pred_tokens = normalize(pred).split()
    gold_tokens = normalize(gold).split()
    
    if not pred_tokens or not gold_tokens:
        return 0.0
    
    # 动态规划计算最长公共子序列
    m, n = len(pred_tokens), len(gold_tokens)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if pred_tokens[i - 1] == gold_tokens[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    
    lcs_len = dp[m][n]
    precision = lcs_len / m if m > 0 else 0.0
    recall = lcs_len / n if n > 0 else 0.0
    
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


# =============================================================================
# 评估接口
# =============================================================================

def evaluate_single(prediction: str, gold_answer: str) -> Dict[str, float]:
    """
    评估单个样本的所有指标
    
    Args:
        prediction: 模型预测的答案
        gold_answer: 黄金标准答案
        
    Returns:
        包含所有指标的字典
    """
    return {
        "contain_acc": 1.0 if contain_acc(prediction, gold_answer) else 0.0,
        "f1_score": token_f1(prediction, gold_answer),
        "rouge_l": rouge_l(prediction, gold_answer),
    }


def evaluate(pairs: Iterable[Tuple[str, str]]) -> Tuple[float, float, float]:
    """
    批量评估（pred, gold）对
    
    Args:
        pairs: (prediction, gold_answer) 的可迭代对象
        
    Returns:
        (contain_acc_avg, f1_avg, rouge_l_avg)
    """
    contain_scores = []
    f1_scores = []
    rouge_scores = []
    
    for pred, gold in pairs:
        contain_scores.append(1.0 if contain_acc(pred, gold) else 0.0)
        f1_scores.append(token_f1(pred, gold))
        rouge_scores.append(rouge_l(pred, gold))
    
    contain_avg = sum(contain_scores) / len(contain_scores) if contain_scores else 0.0
    f1_avg = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0
    rouge_avg = sum(rouge_scores) / len(rouge_scores) if rouge_scores else 0.0
    
    return contain_avg, f1_avg, rouge_avg


def evaluate_dataset(
    dataset_path: str,
    use_rag: bool = True,
    sample_size: int | None = None,
) -> Dict[str, any]:
    """
    评估整个数据集上的RAG系统性能
    
    Args:
        dataset_path: 数据集文件路径
        use_rag: 是否使用RAG系统进行评估
        sample_size: 要评估的样本数量，None表示所有样本
        
    Returns:
        包含所有评估指标的结果字典
    """
    print(f"加载数据集: {dataset_path}")
    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    
    if sample_size:
        dataset = dataset[:sample_size]
    
    print(f"评估 {len(dataset)} 个样本...")
    
    results = {
        "total_samples": len(dataset),
        "samples": [],
        "metrics": {
            "contain_acc": [],
            "f1_score": [],
            "rouge_l": [],
        }
    }
    
    for idx, item in enumerate(dataset):
        question = item.get("question", "")
        gold_answer = item.get("answer", "")
        item_id = item.get("_id", idx)
        
        if not question or not gold_answer:
            print(f"样本 {idx+1}/{len(dataset)}: 跳过（缺少question或answer）")
            continue
        
        try:
            if use_rag:
                # 使用RAG系统生成答案
                prediction, evidence_pool, state = solve_question(question)
            else:
                # 简单的回退：返回答案本身作为演示
                prediction = gold_answer
            
            # 评估指标
            sample_metrics = evaluate_single(prediction, gold_answer)
            
            # 收集结果
            results["samples"].append({
                "id": item_id,
                "question": question,
                "gold_answer": gold_answer,
                "prediction": prediction,
                "metrics": sample_metrics,
            })
            
            # 累积指标
            results["metrics"]["contain_acc"].append(sample_metrics["contain_acc"])
            results["metrics"]["f1_score"].append(sample_metrics["f1_score"])
            results["metrics"]["rouge_l"].append(sample_metrics["rouge_l"])
            
            # 打印进度
            if (idx + 1) % 10 == 0 or idx == len(dataset) - 1:
                print(f"样本 {idx+1}/{len(dataset)}: "
                      f"Contain-ACC={sample_metrics['contain_acc']:.3f}, "
                      f"F1={sample_metrics['f1_score']:.3f}, "
                      f"ROUGE-L={sample_metrics['rouge_l']:.3f}")
        
        except Exception as e:
            print(f"样本 {idx+1}/{len(dataset)}: 评估失败 - {str(e)}")
            continue
    
    # 计算平均指标
    if results["samples"]:
        results["average_metrics"] = {
            "contain_acc": sum(results["metrics"]["contain_acc"]) / len(results["metrics"]["contain_acc"]),
            "f1_score": sum(results["metrics"]["f1_score"]) / len(results["metrics"]["f1_score"]),
            "rouge_l": sum(results["metrics"]["rouge_l"]) / len(results["metrics"]["rouge_l"]),
        }
    else:
        results["average_metrics"] = {
            "contain_acc": 0.0,
            "f1_score": 0.0,
            "rouge_l": 0.0,
        }
    
    return results


def print_evaluation_report(results: Dict[str, any]) -> None:
    """打印评估结果报告"""
    print("\n" + "="*60)
    print("RAG系统性能评估报告")
    print("="*60)
    
    print(f"\n总样本数: {results['total_samples']}")
    print(f"成功评估: {len(results['samples'])}")
    
    if results["samples"]:
        avg_metrics = results["average_metrics"]
        print("\n平均指标:")
        print(f"  Contain-ACC: {avg_metrics['contain_acc']:.4f}")
        print(f"  F1-Score:    {avg_metrics['f1_score']:.4f}")
        print(f"  ROUGE-L:     {avg_metrics['rouge_l']:.4f}")
        
        # 计算统计信息
        contain_scores = results["metrics"]["contain_acc"]
        f1_scores = results["metrics"]["f1_score"]
        rouge_scores = results["metrics"]["rouge_l"]
        
        print("\nContain-ACC 统计:")
        print(f"  最小值: {min(contain_scores):.4f}")
        print(f"  最大值: {max(contain_scores):.4f}")
        print(f"  中位数: {sorted(contain_scores)[len(contain_scores)//2]:.4f}")
        
        print("\nF1-Score 统计:")
        print(f"  最小值: {min(f1_scores):.4f}")
        print(f"  最大值: {max(f1_scores):.4f}")
        print(f"  中位数: {sorted(f1_scores)[len(f1_scores)//2]:.4f}")
        
        print("\nROUGE-L 统计:")
        print(f"  最小值: {min(rouge_scores):.4f}")
        print(f"  最大值: {max(rouge_scores):.4f}")
        print(f"  中位数: {sorted(rouge_scores)[len(rouge_scores)//2]:.4f}")
    
    print("="*60 + "\n")


def save_evaluation_results(results: Dict[str, any], output_path: str) -> None:
    """保存评估结果到JSON文件"""
    # 确保输出目录存在
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"评估结果已保存到: {output_path}")


def save_evaluation_excel(results: Dict[str, any], output_path: str) -> None:
    """保存评估报告到Excel文件"""
    import csv
    
    # 确保输出目录存在
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    if not results["samples"]:
        print("没有评估结果，跳过Excel导出")
        return
    
    # 计算统计信息
    avg = results["average_metrics"]
    contain_scores = results["metrics"]["contain_acc"]
    f1_scores = results["metrics"]["f1_score"]
    rouge_scores = results["metrics"]["rouge_l"]
    
    # 构建报告数据
    report_data = [
        ["指标", "平均值", "最小值", "最大值", "中位数"],
        [
            "Contain-ACC",
            f"{avg['contain_acc']:.4f}",
            f"{min(contain_scores):.4f}",
            f"{max(contain_scores):.4f}",
            f"{sorted(contain_scores)[len(contain_scores)//2]:.4f}"
        ],
        [
            "F1-Score",
            f"{avg['f1_score']:.4f}",
            f"{min(f1_scores):.4f}",
            f"{max(f1_scores):.4f}",
            f"{sorted(f1_scores)[len(f1_scores)//2]:.4f}"
        ],
        [
            "ROUGE-L",
            f"{avg['rouge_l']:.4f}",
            f"{min(rouge_scores):.4f}",
            f"{max(rouge_scores):.4f}",
            f"{sorted(rouge_scores)[len(rouge_scores)//2]:.4f}"
        ],
    ]
    
    # 保存为CSV
    with open(output_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(report_data)
    
    print(f"评估报告已保存到: {output_path}")


def main():
    """主评估函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="评估RAG系统性能")
    parser.add_argument(
        "--dataset",
        type=str,
        default="data/hotpotqa.json",
        help="数据集文件路径 (默认: data/hotpotqa.json)"
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="评估的样本数量 (默认: 所有样本)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output/evaluation_results.json",
        help="输出结果文件路径 (默认: output/evaluation_results.json)"
    )
    parser.add_argument(
        "--use-rag",
        action="store_true",
        default=True,
        help="使用RAG系统评估 (默认: True)"
    )
    
    args = parser.parse_args()
    
    # 执行评估
    results = evaluate_dataset(
        dataset_path=args.dataset,
        use_rag=args.use_rag,
        sample_size=args.sample_size,
    )
    
    # 打印报告
    print_evaluation_report(results)
    
    # 保存结果
    save_evaluation_results(results, args.output)
    
    # 保存Excel报告
    excel_path = args.output.replace(".json", ".csv")
    save_evaluation_excel(results, excel_path)


if __name__ == "__main__":
    main()
