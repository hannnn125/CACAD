from sklearn.metrics import brier_score_loss
import numpy as np
import torch
from typing import Tuple, Dict

def compute_brier_scores(
    all_logits: torch.Tensor,    # [N, num_labels, vocab_size]
    all_labels: torch.Tensor,    # [N, num_labels] (0 또는 1 정수)
    target_token_ids: Tuple[int, int] # (id_for_0, id_for_1)
) -> Dict[str, float]:
    """
    항목별 Brier Score와 전체 평균 Brier Score를 계산합니다.
    """
    id_0, id_1 = target_token_ids
    num_samples, num_labels, _ = all_logits.shape
    
    label_brier_dict = {}
    all_label_scores = []

    # 각 항목(방임, 정서, 신체, 학대)별로 루프
    for l_idx in range(num_labels):
        # 1. 해당 항목의 로짓 추출
        logits_for_label = all_logits[:, l_idx, :] # [N, vocab_size]
        
        # 2. '0'과 '1' 토큰에 대해서만 다시 Softmax 취하기 (이진 분류 확률)
        # 선택된 두 토큰의 로짓만 슬라이싱
        relevant_logits = logits_for_label[:, [id_0, id_1]] # [N, 2]
        probs = torch.softmax(relevant_logits, dim=-1) # [N, 2]
        
        # 3. '1'(학대 있음)일 확률 추출 (Brier Score는 보통 Positive class 확률로 계산)
        prob_positive = probs[:, 1].detach().cpu().numpy()
        true_labels = all_labels[:, l_idx].detach().cpu().numpy()
        
        # 4. 항목별 Brier Score 계산
        score = brier_score_loss(true_labels, prob_positive)
        label_brier_dict[f"label_{l_idx}_brier"] = score
        all_label_scores.append(score)

    # 5. 전체 대상 (Macro Average) 계산
    label_brier_dict["macro_avg_brier"] = np.mean(all_label_scores)
    
    return label_brier_dict