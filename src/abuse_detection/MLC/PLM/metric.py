
import numpy as np
import pandas as pd
import os
import torch
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score,multilabel_confusion_matrix, hamming_loss

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average="macro")
    recall = recall_score(labels, predictions, average="macro")
    precision = precision_score(labels, predictions, average="macro")
    return {
        "accuracy": accuracy,
        "f1": f1,
        "recall": recall,
        "precision": precision,
    }
    
def compute_MultiBERT_metrics(eval_pred):
    logits, labels = eval_pred
    preds = (torch.sigmoid(torch.tensor(logits)) > 0.5).int().numpy()
    labels = labels.astype(int)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="macro"),
        "recall": recall_score(labels, preds, average="macro"),
        "precision": precision_score(labels, preds, average="macro"),
    }


def evaluate_multilabel_detailed(y_true, y_pred, label_names):
    """
    y_true, y_pred: (n_samples, n_labels) 형태의 array-like 데이터.
    label_names: 각 라벨의 이름 리스트 (예: ["방임", "정서", "신체", "성"])
    
    각 라벨에 대해 실제 1과 0 각각의 Accuracy, Precision, Recall, F1-score를 계산
    """
    conf_mats = multilabel_confusion_matrix(y_true, y_pred)
    results = []
    
    for i, label in enumerate(label_names):
        cm = conf_mats[i]  # 해당 라벨의 confusion matrix: [[TN, FP], [FN, TP]]
        tn, fp, fn, tp = cm.ravel()
        total = tn + fp + fn + tp
        # acc = (tn + tp) / (total + 1e-9)  # 전체 Accuracy
        pos_acc = tp / (tp + fn + 1e-9)  # 실제 1 (positive)인 경우
        neg_acc = tn / (tn + fp + 1e-9)  # 실제 0 (negative)인 경우
        
        # 클래스 1 (positive) 평가
        precision_pos = tp / (tp + fp + 1e-9)
        recall_pos = tp / (tp + fn + 1e-9)
        f1_pos = 2 * precision_pos * recall_pos / (precision_pos + recall_pos + 1e-9) if (precision_pos + recall_pos) > 0 else 0
        
        # 클래스 0 (negative) 평가: 0을 positive로 계산하기 위해 역할을 반전
        precision_neg = tn / (tn + fn + 1e-9)
        recall_neg = tn / (tn + fp + 1e-9)
        f1_neg = 2 * precision_neg * recall_neg / (precision_neg + recall_neg + 1e-9) if (precision_neg + recall_neg) > 0 else 0
        
        results.append({
            "Label": label,
            "Actual": 1,
            "Accuracy": round(pos_acc, 4),
            "Precision": round(precision_pos, 4),
            "Recall": round(recall_pos, 4),
            "F1-score": round(f1_pos, 4)
        })
        results.append({
            "Label": label,
            "Actual": 0,
            "Accuracy": round(neg_acc, 4),
            "Precision": round(precision_neg, 4),
            "Recall": round(recall_neg, 4),
            "F1-score": round(f1_neg, 4)
        })
        
    return pd.DataFrame(results)


def evaluate_and_save_metrics(test_labels,
                              test_preds,
                              save_dir,
                              label_names=["방임", "정서", "신체", "성"],
                              file_names=None):
    """
    멀티라벨 분류 평가 결과를 출력하고, 파일로 저장

    Args:
        test_labels (np.ndarray): 실제 정답 (binary multi-label)
        test_preds  (np.ndarray): 예측 결과 (binary multi-label)
        save_dir    (str): 평가 결과를 저장할 디렉토리
        label_names (list): 라벨 이름 리스트
        file_names  (list): 각 샘플의 원본 JSON 파일 이름 리스트 (optional)
    """
    # Exact Match Accuracy 계산
    exact_match_acc = accuracy_score(test_labels, test_preds)
    macro_f1        = f1_score(test_labels, test_preds, average="macro")
    micro_f1        = f1_score(test_labels, test_preds, average="micro")
    macro_precision= precision_score(test_labels, test_preds, average="macro")
    macro_recall   = recall_score(test_labels, test_preds, average="macro")

    # detailed per-label metrics (기존 구현 그대로 사용)
    df_metrics = evaluate_multilabel_detailed(test_labels, test_preds, label_names=label_names)

    # 콘솔과 텍스트 리포트 저장 (생략)
    # 출력
    print(f"\nExact Match Accuracy: {exact_match_acc:.4f}")
    print(f"Macro Precision: {macro_precision:.4f}")
    print(f"Macro Recall: {macro_recall:.4f}")
    print(f"Macro F1 Score: {macro_f1:.4f}")
    print(f"Micro F1 Score: {micro_f1:.4f}")
    print(df_metrics)

    # 저장
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, "test_report.txt"), "w", encoding="utf-8") as f:
        f.write(f"Exact Match Accuracy: {exact_match_acc:.4f}\n")
        f.write(f"Macro Precision: {macro_precision:.4f}\n")
        f.write(f"Macro Recall: {macro_recall:.4f}\n")
        f.write(f"Macro F1 Score: {macro_f1:.4f}\n")
        f.write(f"Micro F1 Score: {micro_f1:.4f}\n")
        f.write("\nDetailed Metrics (per label):\n")
        f.write(df_metrics.to_string())

    #── 예측 결과 저장 ─────────────────────────────────────────────
    df_true = pd.DataFrame(test_labels, columns=[f"true_{n}"    for n in label_names])
    df_pred = pd.DataFrame(test_preds,  columns=[f"pred_{n}"    for n in label_names])
    df_corr = pd.DataFrame((df_true.values == df_pred.values).astype(int),
                           columns=[f"correct_{n}" for n in label_names])
    df_exact= pd.DataFrame({
        "exact_match": df_corr.all(axis=1).astype(int)
    })

    # 원본 순서대로 합치기
    df_combined = pd.concat([df_true, df_pred, df_corr, df_exact], axis=1)

    # 파일 이름이 주어졌으면 첫 컬럼으로 삽입
    if file_names is not None:
        if len(file_names) != len(df_combined):
            raise ValueError("file_names 길이가 샘플 수와 일치하지 않습니다")
        df_combined.insert(0, "file_name", file_names)

    os.makedirs(save_dir, exist_ok=True)
    out_csv = os.path.join(save_dir, "predictions.csv")
    df_combined.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"Saved detailed predictions to {out_csv}")
    
def evaluate_metrics(test_labels,
                     test_preds,
                     label_names=["방임", "정서", "신체", "성"]):
    """
    멀티라벨 분류 평가 결과를 계산하여 리턴

    Args:
        test_labels (np.ndarray): 실제 정답 (binary multi-label)
        test_preds  (np.ndarray): 예측 결과 (binary multi-label)
        label_names (list): 라벨 이름 리스트

    Returns:
        tuple: (exact_match_acc, macro_precision, macro_recall, macro_f1, micro_f1)
    """
    # 정확도·정밀도·재현율·F1 계산
    exact_match_acc = accuracy_score(test_labels, test_preds)
    macro_precision= precision_score(test_labels, test_preds, average="macro")
    macro_recall   = recall_score(test_labels, test_preds, average="macro")
    macro_f1       = f1_score(test_labels, test_preds, average="macro")
    micro_f1       = f1_score(test_labels, test_preds, average="micro")

    return exact_match_acc, macro_precision, macro_recall, macro_f1, micro_f1