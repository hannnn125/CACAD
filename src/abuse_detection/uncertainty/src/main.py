
import argparse
import json
import csv
import os
import numpy as np
import torch
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from tqdm import tqdm

# from utils.ECE import compute_multilabel_ece
# from utils.Brier import compute_multilabel_brier
from utils.uncertainty import compute_MSP, compute_entropy, compute_mismatch


def generate_with_uncertainty(model, tokenizer, device, text, max_tokens=None):

    prompt = text + "<|pred|>"
    inputs = tokenizer(prompt, return_tensors='pt', return_token_type_ids=False).to(device)
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']

    # 0/1 토큰 아이디 후보 수집 (모델/토크나이저별 변형 대응)
    one_token_ids = set()
    zero_token_ids = set()
    for token_str in ["1", " 1"]:
        token_ids = tokenizer.encode(token_str, add_special_tokens=False)
        if len(token_ids) == 1:
            one_token_ids.add(token_ids[0])
    for token_str in ["0", " 0"]:
        token_ids = tokenizer.encode(token_str, add_special_tokens=False)
        if len(token_ids) == 1:
            zero_token_ids.add(token_ids[0])

    generated_ids = []
    step_entropies = []
    step_msps = []
    step_probs_1 = []  # "1" 확률

    """
    0,1이 아닌 값으로 뱉는 경우 모두 0으로 변환하므로 1 토큰 확률만 저장
    """

    for _ in range(max_tokens):
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            next_token_logits = outputs.logits[0, -1, :]
            
            # 1. 확률 계산
            probs = torch.softmax(next_token_logits, dim=-1)

            # 1-1 토큰 확률 저장 (가능한 "1" 토큰 id 합산)
            if one_token_ids:
                probs_1 = float(sum(probs[token_id].item() for token_id in one_token_ids))
            else:
                probs_1 = 0.0
            
            # 2. Uncertainty 지표 계산 
            probs_np = probs.detach().cpu().numpy()
            entropy_val = compute_entropy(probs_np)
            msp_val = compute_MSP(probs_np)
            
            # 3. 다음 토큰 선택 (Argmax)
            next_token_id = int(torch.argmax(probs).item())
            
            # 4. 데이터 저장 (숫자 토큰만 기록)
            generated_ids.append(next_token_id)
            token_text = tokenizer.decode([next_token_id], skip_special_tokens=True).strip()
            if token_text in {"0", "1"} or next_token_id in one_token_ids or next_token_id in zero_token_ids:
                step_entropies.append(round(entropy_val, 4))
                step_msps.append(round(msp_val, 4))
                step_probs_1.append(round(probs_1, 4))

            # 5. 다음 입력을 위해 업데이트[]
            next_token_tensor = torch.tensor([[next_token_id]]).to(device)
            input_ids = torch.cat([input_ids, next_token_tensor], dim=1)
            attention_mask = torch.cat(
                [attention_mask, torch.ones((1, 1), dtype=attention_mask.dtype).to(device)], dim=1)

            if next_token_id == tokenizer.eos_token_id:
                break
    # 최종 예측 (0/1) - 생성 텍스트 기반 파싱
    generated_part = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    tokens = generated_part.split()


    preds = []
    for token in tokens:
        try:
            preds.append(int(token))
        except ValueError:
            preds.append(0)
    preds = preds
    probs = step_probs_1
    entropies = step_entropies
    msps = step_msps
    return preds, step_probs_1, entropies, msps


def run_test(checkpoint_path, save_dir, test_data, max_tokens):
    # 1. 모델 및 토크나이저 로드
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = AutoConfig.from_pretrained(checkpoint_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(checkpoint_path, config=config, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path, trust_remote_code=True)
    
    if len(tokenizer) > model.get_input_embeddings().weight.shape[0]:
        model.resize_token_embeddings(len(tokenizer))

    model.to(device)
    model.eval()

    # 2. 데이터 로드
    raw_datasets = load_dataset("json", data_files={"test": test_data})
    
    all_labels = []
    all_preds = []
    all_probs = []
    all_entropies = []
    all_msps = []
    all_mismatches_entropy = []
    all_mismatches_msp = []

    # 3. 추론 루프
    for data in tqdm(raw_datasets['test']):
        label = data['pred'] # [0, 1, 0, 0] 형태
        preds, probs, entropies, msps = generate_with_uncertainty(
            model,
            tokenizer,
            device,
            data['text'],
            max_tokens=max_tokens,
        )
        mismatch = compute_mismatch(label, preds, entropies, msps)

        """
        형태 예시
        --------------------------------
        label:  [0, 1, 0, 0]
        preds:  [0, 1, 0, 0]
        probs:  [0.0124, 0.9852, 0.0421, 0.0089]
        entropies:  [0.1245, 0.0521, 0.2142, 0.0982]
        msps:  [0.9821, 0.9852, 0.9432, 0.9912]
        --------------------------------
        """

        all_labels.append(label)
        all_preds.append(preds)
        all_probs.append(probs)
        all_entropies.append(entropies)
        all_msps.append(msps)
        all_mismatches_entropy.append(mismatch["entropy"])
        all_mismatches_msp.append(mismatch["msp"])

    # 4. 결과 집계 (Numpy 변환)
    all_labels_np = np.array(all_labels)
    all_probs_np = np.array(all_probs)

    # 5. 항목별 지표 계산
    metrics_report = {}

    # 7. 결과 저장 (CSV)
    save_uncertainty_results(
        save_dir=save_dir,
        all_labels=all_labels,
        all_preds=all_preds,
        all_entropies=all_entropies,
        all_msps=all_msps,
        all_mismatches_entropy=all_mismatches_entropy,
        all_mismatches_msp=all_mismatches_msp,
    )

def save_uncertainty_results(save_dir, all_labels, all_preds, all_entropies, all_msps, all_mismatches_entropy, all_mismatches_msp):
    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir,  "test_results_entropy.csv")
    fieldnames = ["idx", "ground_truth", "predicted", "entropy", "msp", "mismatch_entropy", "mismatch_msp"]

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, quoting=csv.QUOTE_MINIMAL)
        writer.writeheader()
        for idx, (label, pred, entropy, msp) in enumerate(
            zip(all_labels, all_preds, all_entropies, all_msps)
        ):
            writer.writerow(
                {
                    "idx": idx,
                    "ground_truth": json.dumps(label, ensure_ascii=False),
                    "predicted": json.dumps(pred, ensure_ascii=False),
                    "entropy": json.dumps(entropy, ensure_ascii=False),
                    "msp": json.dumps(msp, ensure_ascii=False),
                    "mismatch_entropy": json.dumps(all_mismatches_entropy[idx], ensure_ascii=False),
                    "mismatch_msp": json.dumps(all_mismatches_msp[idx], ensure_ascii=False),
                }
            )
    print(f"[Saved] {out_path}")

def save_calibration_results(save_dir, metrics_report):
    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir,"calibration_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(metrics_report, f, ensure_ascii=False, indent=2)
    print(f"[Saved] {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_dir", default="/nas/user10_1/Counsell/log/Qwen2.5/250413_Qwen/Qwen2.5-3B_base_ep(20)_lr(1e-5)_batch(2)_alpha(0.1)/checkpoint-1280", type=str)
    parser.add_argument("--base_save_dir", default="/home/user10/vscode/counsell/counsell_new/MLC/model/LLM/QWEN2.5-3B_1e-5/ckpt-1280", type=str)
    # parser.add_argument("--ckpt_dir", default="/nas/counsell/log/polyglot-5.8b/250413_lr(1e-5)_batch(4)_alpha(0.1)/checkpoint-5120", type=str)
    # parser.add_argument("--base_save_dir", default="/home/user10/vscode/counsell/counsell_new/MLC/uncertainty/output/polyglot-ko-5.8b/ckpt-5120", type=str)
    parser.add_argument("--test_data", default="./data/processed/finetuning/test.json", type=str)
    parser.add_argument("--max_tokens", type=int, default=7, help="Qwen: 7, polyglot: 4")
    args = parser.parse_args()

    run_test(args.ckpt_dir, args.base_save_dir, args.test_data, args.max_tokens)    