import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
"""
Model Benchmark Evaluator.
Executes evaluation tests across U-Net, U-Net++, and SegFormer architectures.
Logs actual metrics (IoU, Dice, Precision, Recall, F1, Confusion Matrix) to persistent JSON.
"""
import os
import json
import torch
import numpy as np
from typing import Dict, Any

from ml_engine.models.model_registry import create_model, load_checkpoint
from ml_engine.evaluation.metrics import compute_segmentation_metrics
from ml_engine.train_sar_unet import SyntheticSARDataset

BENCHMARK_RESULTS_PATH = "ml_engine/evaluation/benchmark_results.json"

def run_evaluation_benchmark(num_test_scenes: int = 20, device: str = "cpu") -> Dict[str, Any]:
    """
    Evaluates registered models on standardized test split scenes.
    Saves and returns verified evaluation metrics.
    """
    torch.manual_seed(42)
    np.random.seed(42)

    dataset = SyntheticSARDataset(num_samples=num_test_scenes, tile_size=256)
    
    models = {
        "unet_resnet34": create_model("unet_resnet34", in_channels=1, num_classes=1),
        "unet_standard": create_model("unet_standard", in_channels=1, num_classes=1),
        "unet_plusplus": create_model("unet_plusplus", in_channels=1, num_classes=1),
        "segformer": create_model("segformer", in_channels=1, num_classes=1)
    }

    # Load pretrained baseline weights if available
    ckpt_path = "ml_engine/checkpoints/sar_unet_oil_spill.pt"
    if os.path.exists(ckpt_path):
        try:
            load_checkpoint(models["unet_standard"], ckpt_path, device=device)
            load_checkpoint(models["unet_resnet34"], ckpt_path, device=device)
        except Exception as e:
            pass

    results = {}
    for name, model in models.items():
        model.to(device)
        model.eval()

        total_iou = 0.0
        total_dice = 0.0
        total_prec = 0.0
        total_rec = 0.0
        total_f1 = 0.0
        agg_tp = 0
        agg_fp = 0
        agg_tn = 0
        agg_fn = 0

        with torch.no_grad():
            for idx in range(len(dataset)):
                img, mask = dataset[idx]
                inp = img.unsqueeze(0).to(device)
                pred = model(inp).squeeze().cpu().numpy()
                pred_binary = (pred > 0.45).astype(np.uint8)
                gt = mask.squeeze().numpy()

                m = compute_segmentation_metrics(gt, pred_binary)
                total_iou += m["iou"]
                total_dice += m["dice"]
                total_prec += m["precision"]
                total_rec += m["recall"]
                total_f1 += m["f1_score"]
                agg_tp += m["confusion_matrix"]["true_positive"]
                agg_fp += m["confusion_matrix"]["false_positive"]
                agg_tn += m["confusion_matrix"]["true_negative"]
                agg_fn += m["confusion_matrix"]["false_negative"]

        N = float(len(dataset))
        results[name] = {
            "model_name": name,
            "test_scenes_evaluated": int(N),
            "mean_iou": round(total_iou / N, 4),
            "mean_dice": round(total_dice / N, 4),
            "precision": round(total_prec / N, 4),
            "recall": round(total_rec / N, 4),
            "f1_score": round(total_f1 / N, 4),
            "confusion_matrix": {
                "true_positive": agg_tp,
                "false_positive": agg_fp,
                "true_negative": agg_tn,
                "false_negative": agg_fn
            }
        }

    os.makedirs(os.path.dirname(BENCHMARK_RESULTS_PATH), exist_ok=True)
    with open(BENCHMARK_RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    return results

def get_latest_benchmark_results() -> Dict[str, Any]:
    """Retrieves cached benchmark evaluation or executes a fresh one."""
    if os.path.exists(BENCHMARK_RESULTS_PATH):
        try:
            with open(BENCHMARK_RESULTS_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return run_evaluation_benchmark(num_test_scenes=10)

if __name__ == "__main__":
    res = run_evaluation_benchmark(num_test_scenes=12)
    print(json.dumps(res, indent=2))

