"""
Evaluation Metrics for SAR Segmentation and Classification.
Calculates IoU (Jaccard Index), Dice coefficient, Precision, Recall, F1, and Confusion Matrix.
"""
import numpy as np
from typing import Dict, Any, Tuple

def compute_segmentation_metrics(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-6) -> Dict[str, Any]:
    """
    Computes rigorous segmentation metrics from binary ground truth and predicted arrays.
    y_true: boolean or 0/1 array
    y_pred: boolean or 0/1 array
    """
    y_t = (y_true > 0.5).astype(np.uint8).flatten()
    y_p = (y_pred > 0.5).astype(np.uint8).flatten()

    tp = int(np.sum((y_t == 1) & (y_p == 1)))
    fp = int(np.sum((y_t == 0) & (y_p == 1)))
    fn = int(np.sum((y_t == 1) & (y_p == 0)))
    tn = int(np.sum((y_t == 0) & (y_p == 0)))

    intersection = tp
    union = tp + fp + fn

    iou = float(intersection / (union + eps))
    dice = float((2.0 * tp) / (2.0 * tp + fp + fn + eps))
    precision = float(tp / (tp + fp + eps))
    recall = float(tp / (tp + fn + eps))
    f1 = float(2.0 * (precision * recall) / (precision + recall + eps))
    accuracy = float((tp + tn) / (tp + tn + fp + fn + eps))

    return {
        "iou": round(iou, 4),
        "dice": round(dice, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
        "accuracy": round(accuracy, 4),
        "confusion_matrix": {
            "true_positive": tp,
            "false_positive": fp,
            "true_negative": tn,
            "false_negative": fn
        }
    }

def compute_classification_metrics(y_true_labels: list, y_pred_labels: list, classes: list = None) -> Dict[str, Any]:
    """Computes multi-class classification confusion matrix for OIL, LOOK-ALIKE, UNCERTAIN."""
    if classes is None:
        classes = ["OIL", "LOOK-ALIKE", "UNCERTAIN"]
    
    matrix = {c_true: {c_pred: 0 for c_pred in classes} for c_true in classes}
    for yt, yp in zip(y_true_labels, y_pred_labels):
        if yt in matrix and yp in matrix[yt]:
            matrix[yt][yp] += 1
            
    total = len(y_true_labels)
    correct = sum(matrix[c][c] for c in classes)
    accuracy = round(correct / max(1, total), 4)

    return {
        "classes": classes,
        "confusion_matrix": matrix,
        "overall_accuracy": accuracy,
        "sample_count": total
    }
