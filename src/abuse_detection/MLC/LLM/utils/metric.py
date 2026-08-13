
import os
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def compute_metrics(eval_preds):
    import numpy as np
    predictions, labels = eval_preds
    # 이미 preprocess_logits_for_metrics에서 argmax 적용되어 있음
    predictions = np.array(predictions)
    labels = np.array(labels)
    mask = labels != -100
    if mask.sum() == 0:
        accuracy = 0.0
    else:
        accuracy = (predictions[mask] == labels[mask]).mean()
    return {"accuracy": accuracy}

def evaluate_and_save_metrics(test_labels,
                              test_preds,
                              save_dir,
                              label_names=["방임", "정서", "신체", "성"],
                              file_names=None):
    """
    멀티라벨 분류 평가 결과를 출력하고, 파일로 저장합니다.

    Args:
        test_labels (np.ndarray): 실제 정답 (binary multi-label)
        test_preds  (np.ndarray): 예측 결과 (binary multi-label)
        save_dir    (str): 평가 결과를 저장할 디렉토리
        label_names (list): 라벨 이름 리스트
        file_names  (list): 각 샘플의 원본 JSON 파일 이름 리스트 (optional)
    """
    # 🎯 Exact Match Accuracy 계산
    exact_match_acc = accuracy_score(test_labels, test_preds)
    macro_f1        = f1_score(test_labels, test_preds, average="macro")
    micro_f1        = f1_score(test_labels, test_preds, average="micro")
    macro_precision= precision_score(test_labels, test_preds, average="macro")
    macro_recall   = recall_score(test_labels, test_preds, average="macro")

    # detailed per-label metrics (기존 구현 그대로 사용)
    df_metrics = evaluate_per_label_binary(
    test_labels,
    test_preds,
    label_names=label_names
)

    # 콘솔과 텍스트 리포트 저장 (생략)
    # 출력
    print(f"\n🎯 Exact Match Accuracy: {exact_match_acc:.4f}")
    print(f"Macro Precision: {macro_precision:.4f}")
    print(f"Macro Recall: {macro_recall:.4f}")
    print(f"Macro F1 Score: {macro_f1:.4f}")
    print(f"Micro F1 Score: {micro_f1:.4f}")
    print(df_metrics)

    # 저장
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, "test_report.txt"), "w", encoding="utf-8") as f:
        f.write(f"🎯 Exact Match Accuracy: {exact_match_acc:.4f}\n")
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
    print(f"✅ Saved detailed predictions to {out_csv}")
    

def evaluate_per_label_binary(
    test_labels: np.ndarray,
    test_preds: np.ndarray,
    label_names=["방임", "정서", "신체", "성"]
):
    """
    각 라벨을 단일 이진 분류로 보고
    Acc / Prec / Rec / F1 계산
    """
    results = []

    for i, label in enumerate(label_names):
        y_true = test_labels[:, i]
        y_pred = test_preds[:, i]
        acc  = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec  = recall_score(y_true, y_pred, zero_division=0)
        f1   = f1_score(y_true, y_pred, zero_division=0)

        results.append({
            "Category": label,
            "Acc": round(acc, 4),
            "Prec": round(prec, 4),
            "Rec": round(rec, 4),
            "F1": round(f1, 4)
        })
    
    return pd.DataFrame(results)