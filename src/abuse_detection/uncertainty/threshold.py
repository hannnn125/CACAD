import pandas as pd
import numpy as np
import os
import sys
import json
from sklearn.metrics import f1_score

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def run_threshold_analysis(csv_path, results_path, metric_name='entropy', thresholds=None):
    if thresholds is None:
        if metric_name == 'entropy':
            thresholds = np.arange(0.0, 0.7005, 0.0005)
        else: # msp
            thresholds = np.arange(0.5005, 1.0005, 0.0005)

    df = pd.read_csv(csv_path)
    
    # 데이터 파싱
    for col in ['ground_truth', 'predicted', metric_name]:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: json.loads(x) if isinstance(x, str) else x)

    total_rows = len(df)
    all_results = []

    for threshold in thresholds:
        pending_count = 0  # 상담사에게 넘어간 건수
        non_pending_count = 0
        final_preds = []
        final_truths = []

        for i in df.index:
            truths = df.at[i, 'ground_truth']
            preds = df.at[i, 'predicted']
            uncertainty = df.at[i, metric_name]
            
            if metric_name == 'entropy':
                uncertainty = max(uncertainty)
                # 엔트로피가 임계값보다 높으면 보류(Pending)
                is_pending = uncertainty > threshold
            else: # msp
                uncertainty = min(uncertainty)
                # MSP가 임계값보다 낮으면  보류(Pending)
                is_pending = uncertainty < threshold
            
            if is_pending:
                pending_count += 1
                final_preds.append(truths)
            else:
                if preds == truths:
                    non_pending_count += 1
                final_preds.append(preds)
            final_truths.append(truths)

        # 통계 계산
        pending_ratio = pending_count / total_rows
        ai_acc_term = non_pending_count / total_rows
        
        # 상담사 정확도별 시스템 전체 성능 (Exact Match Accuracy)
        acc_h100 = ai_acc_term + (1.0 * pending_ratio)
        acc_h90  = ai_acc_term + (0.9 * pending_ratio)
        acc_h80  = ai_acc_term + (0.8 * pending_ratio)

        macro_f1_human_100 = f1_score(final_truths, final_preds, average='macro', zero_division=0)

        all_results.append({
            'threshold': round(threshold, 4),
            'pending_count': pending_count,
            'pending_ratio': round(pending_ratio, 4),
            'exact_match_acc_human_100%': round(acc_h100, 4),
            'exact_match_acc_human_90%': round(acc_h90, 4),
            'exact_match_acc_human_80%': round(acc_h80, 4),
            'macro_f1_human_100%': round(macro_f1_human_100, 4)
        })

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(results_path, index=False)
    print(f"완료: [{metric_name}] 분석 결과가 {results_path} 에 저장되었습니다.")

if __name__ == "__main__":
    # polyglot결과 경로
    # base_dir = "outputs/uncertainty/polyglot-ko-5.8b/ckpt-5120"
    # QWEN 결과 경로
    base_dir = "outputs/uncertainty/Qwen2.5-3B-Instruct/ckpt-1280"
    csv_path = os.path.join(base_dir, "test_results_entropy.csv")
    
    run_threshold_analysis(csv_path, os.path.join(base_dir, "threshold_performance_entropy.csv"), metric_name='entropy')
    run_threshold_analysis(csv_path, os.path.join(base_dir, "threshold_performance_msp.csv"), metric_name='msp')
