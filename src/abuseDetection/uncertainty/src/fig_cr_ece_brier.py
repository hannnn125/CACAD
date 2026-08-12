import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import os
import torch
from sklearn.metrics import brier_score_loss

def plot_coverage_risk_refined(csv_path, save_dir):
    df = pd.read_csv(csv_path)
    
    # 1. 파일 단위 정답 여부 판단
    df['gt'] = df['ground_truth'].apply(json.loads)
    df['pred'] = df['predicted'].apply(json.loads)
    df['is_error'] = df.apply(lambda row: 1 if row['gt'] != row['pred'] else 0, axis=1)

    # 2. 통합 확신도(Confidence) 설정
    def get_final_conf(row):
        msps = json.loads(row['msp'])
        return min(msps)

    df['file_conf'] = df.apply(get_final_conf, axis=1)

    # 3. 데이터 정렬 및 Risk Curve 계산
    df_sorted = df.sort_values(by='file_conf', ascending=False)
    errors = df_sorted['is_error'].values
    n = len(errors)
    coverage = np.arange(1, n + 1) / n
    risk = np.cumsum(errors) / np.arange(1, n + 1)

    aurc = np.trapz(risk, coverage)

    # 4. 시각화
    plt.figure(figsize=(10, 6))
    plt.plot(coverage, risk, color='blue', lw=2, label='File-level risk curve')
    plt.axhline(y=errors.mean(), color='red', linestyle='--', label='Baseline error')
    plt.xlabel('Coverage')
    plt.ylabel('Risk')
    # plt.title(f'Qwen2.5-3B-Instruct coverage-risk curve\nAURC: {aurc:.3f}')
    plt.title(f'Polyglot-Ko-5.8B coverage-risk curve\nAURC: {aurc:.3f}')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig(os.path.join(save_dir, "Poly_system_coverage_risk1.png"))
    plt.close()

    print(f"Analysis Complete. AURC: {aurc:.4f}")
    return df 



def get_bin_statistics(df, bins):
    """커스텀 빈에 맞춰 Accuracy, Confidence, Count를 계산하는 공통 함수"""
    confidences = df['file_conf'].values
    accuracies_binary = 1 - df['is_error'].values
    
    bin_accs = []
    bin_confs = []
    bin_counts = []
    
    for i in range(len(bins) - 1):
        lower, upper = bins[i], bins[i+1]
        # 마지막 bin은 양끝 포함, 나머지는 [lower, upper)
        if i == len(bins) - 2:
            mask = (confidences >= lower) & (confidences <= upper)
        else:
            mask = (confidences >= lower) & (confidences < upper)
            
        if mask.any():
            bin_accs.append(accuracies_binary[mask].mean())
            bin_confs.append(confidences[mask].mean())
            bin_counts.append(mask.sum())
        else:
            # 데이터가 없는 빈은 0으로 채우되 계산 시 가중치로 인해 무시됨
            bin_accs.append(0.0)
            bin_confs.append(0.0)
            bin_counts.append(0)
            
    return np.array(bin_accs), np.array(bin_confs), np.array(bin_counts)

def plot_calibration_curve_style_custom(df, save_dir, bins):
    # 1. 시각화 데이터 준비
    bin_accs, bin_confs, bin_counts = get_bin_statistics(df, bins)
    
    # 데이터가 존재하는 빈만 선택
    valid_indices = bin_counts > 0
    plot_confs = bin_confs[valid_indices]
    plot_accs = bin_accs[valid_indices]

    # 2. 메트릭 계산 (수동 ECE 및 Brier)
    total_count = np.sum(bin_counts)
    abs_diff = np.abs(bin_accs - bin_confs)
    manual_ece = np.sum(abs_diff * bin_counts) / total_count
    
    y_true = 1 - df['is_error'].values
    y_prob = df['file_conf'].values
    brier = brier_score_loss(y_true, y_prob)

    # 3. 시각화 시작
    plt.figure(figsize=(7, 7))
    
    # 가이드라인 (Perfectly calibrated)
    plt.plot([0, 1], [0, 1], "k--", alpha=0.8, label="Perfectly calibrated")
    
    # Calibration Curve (보라색 스타일)
    plt.plot(
        plot_confs,
        plot_accs,
        "o-",
        color="#BC59CF",
        linewidth=2.5,
        markersize=6,
        mfc="white",
        mec="#BC59CF",
        mew=2,
        label="Calibration curve",
    )
    
    # 축 범위 설정 (0부터 1까지 전체 표시)
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.02])
    
    # 스타일링: 그리드 및 테두리 투명도 조정
    plt.grid(True, linestyle='--', alpha=0.5)
    # plt.gca().spines['top'].set_visible(False)
    # plt.gca().spines['right'].set_visible(False)

    plt.xlabel("Mean Predicted Probability", fontsize=11)
    plt.ylabel("True Positive Rate", fontsize=11)
    
    # 제목 및 메트릭 표시 (이미지 스타일처럼 상단 배치)
    title_str = "Reliability Diagram" # 또는 모델명
    plt.title(f"{title_str}\nBrier: {brier:.4f} | ECE: {manual_ece:.4f}", 
              fontsize=12, pad=15)
    
    plt.legend(loc="upper left", frameon=True)
    plt.tight_layout()
    
    # 저장
    save_path = os.path.join(save_dir, "calibration_curve_refined.png")
    plt.savefig(save_path, dpi=300)
    plt.close()

    return manual_ece
# # --- 실행 ---
# csv_path = "/Users/hyunyounge00/Documents/VScode/counsell_github/src/detection/uncertainty/output/QWEN2.5-3B_1e-5/ckpt-1280/test_results_entropy.csv"
# save_dir = "/Users/hyunyounge00/Documents/VScode/counsell_github/src/detection/uncertainty/output/QWEN2.5-3B_1e-5/ckpt-1280"

csv_path = "/Users/hyunyounge00/Documents/VScode/counsell_github/src/detection/uncertainty/output/polyglot-ko-5.8b/ckpt-5120/test_results_entropy.csv"
save_dir = "/Users/hyunyounge00/Documents/VScode/counsell_github/src/detection/uncertainty/output/polyglot-ko-5.8b/ckpt-5120"

processed_df = plot_coverage_risk_refined(csv_path, save_dir)
CUSTOM_BINS = np.quantile(processed_df['file_conf'], np.linspace(0, 1, 11))
ece_val = plot_calibration_curve_style_custom(processed_df, save_dir, CUSTOM_BINS)
print(f"Calculated ECE: {ece_val:.4f}")