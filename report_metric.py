import argparse
import os
import json
import re
import nltk
import numpy as np
from tqdm import tqdm
from rouge import Rouge
from nltk.translate.meteor_score import single_meteor_score
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# 下载 NLTK 资源（只需一次）
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)


def prep_reports(reports):
    """Preprocesses reports"""
    return [list(filter(
        lambda val: val != "", str(elem).lower().replace(".", " .").split(" ")
    )) for elem in reports]


def main(args):
    rouge_scorer = Rouge()
    smoothie = SmoothingFunction().method1

    results = []

    datas = range(args.start_id, args.end_id)
    for i in tqdm(datas):
        response_path = os.path.join(args.response_dir, f"text_{i}.txt")
        gt_path = os.path.join(args.gt_dir, f"text_{i}.txt")

        if not os.path.exists(response_path) or not os.path.exists(gt_path):
            continue

        with open(response_path, "r") as f:
            response = f.read().strip()

        with open(gt_path, "r") as f:
            golden = f.read().strip()

        if response == "":
            continue

        tokenized_response = prep_reports([response])[0]
        tokenized_golden = prep_reports([golden])[0]

        bleu1 = sentence_bleu([tokenized_golden], tokenized_response, weights=[1], smoothing_function=smoothie)
        bleu2 = sentence_bleu([tokenized_golden], tokenized_response, weights=[0.5, 0.5], smoothing_function=smoothie)
        bleu3 = sentence_bleu([tokenized_golden], tokenized_response, weights=[1/3]*3, smoothing_function=smoothie)
        bleu4 = sentence_bleu([tokenized_golden], tokenized_response, weights=[0.25]*4, smoothing_function=smoothie)

        try:
            rouge_scores = rouge_scorer.get_scores(response.lower(), golden.lower())[0]
        except Exception:
            continue

        meteor = single_meteor_score(hypothesis=tokenized_response, reference=tokenized_golden)

        results.append({
            "id": i,
            "bleu1": bleu1,
            "bleu2": bleu2,
            "bleu3": bleu3,
            "bleu4": bleu4,
            "rouge1": rouge_scores["rouge-1"]["f"],
            "rouge2": rouge_scores["rouge-2"]["f"],
            "rougeL": rouge_scores["rouge-l"]["f"],
            "meteor": meteor,
        })

    # 计算平均分
    metrics = {k: np.mean([r[k] for r in results]) for k in results[0].keys() if k != "id"}

    print("metrics:")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

    # 保存到文件
    with open(args.output_metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nSaved metrics to: {args.output_metrics_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute BLEU, ROUGE, METEOR metrics for report generation.")
    parser.add_argument("--response_dir", type=str, required=True, help="Directory containing generated reports")
    parser.add_argument("--gt_dir", type=str, required=True, help="Directory containing ground-truth reports")
    parser.add_argument("--output_metrics_path", type=str, required=True, help="Path to save metrics JSON")
    parser.add_argument("--start_id", type=int, default=0, help="Start index for evaluation")
    parser.add_argument("--end_id", type=int, default=5000, help="End index for evaluation")
    args = parser.parse_args()
    main(args)
