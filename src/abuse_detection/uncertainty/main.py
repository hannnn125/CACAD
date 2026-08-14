import os
import json
import csv
import argparse
import numpy as np
from datasets import load_dataset
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
import torch
from tqdm import tqdm
from utils.uncertainty import compute_entropy, compute_mismatch, compute_MSP

def generate_with_uncertainty(model, tokenizer, device, text, max_tokens=None):
    prompt = text + "<|pred|>"
    inputs = tokenizer(prompt, return_tensors='pt', return_token_type_ids=False).to(device)
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']
    
    zero_token_ids = tokenizer.encode("0", add_special_tokens=False)[0]
    one_token_ids = tokenizer.encode("1", add_special_tokens=False)[0]

    generated_ids = []
    step_entropies = []
    step_msps = []
    step_probs_1 = []

    for _ in range(max_tokens):
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            next_token_logits = outputs.logits[0,-1, :]

            probs = torch.softmax(next_token_logits, dim=-1)
            p0 = probs[zero_token_ids]
            p1 = probs[one_token_ids]
            p0_hat = p0 / (p0 + p1)
            p1_hat = p1 / (p0 + p1)

            probs01 = np.array([p0_hat, p1_hat])
            entropy = compute_entropy(probs01)  # 길이 2
            msp = compute_MSP(probs01)

            next_token_id = int(torch.argmax(probs).item())
            generated_ids.append(next_token_id)
            token_text = tokenizer.decode([next_token_id], skip_special_tokens=True).strip()
            if token_text in {"0", "1"} or next_token_id in one_token_ids or next_token_id in zero_token_ids:
                step_entropies.append(round(entropy, 4))
                step_msps.append(round(msp, 4))
                step_probs_1.append(round(p1_hat, 4))
            next_token_tensor = torch.tensor([[next_token_id]]).to(device)
            input_ids = torch.cat([input_ids, next_token_tensor], dim=1)
            attention_mask = torch.cat([attention_mask, torch.ones((1, 1), dtype=attention_mask.dtype).to(device)], dim=1)
            
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
    entropies = step_entropies
    msps = step_msps
    probs = step_probs_1
    return preds, probs, entropies, msps

def run_test(ckpt_dir, save_dir, test_data, max_tokens):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = AutoConfig.from_pretrained(ckpt_dir, trust_remote_code=True)
    # 모델 및 토크나이저 로드
    model = AutoModelForCausalLM.from_pretrained(ckpt_dir, config=config, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(ckpt_dir, trust_remote_code=True)

    if len(tokenizer) > model.get_input_embeddings().weight.shape[0]:
        model.resize_token_embeddings(len(tokenizer))
    model.to(device)
    model.eval()

    # 테스트 데이터 로드
    raw_datasets = load_dataset("json", data_files={"test": test_data})
    
    all_labels = []
    all_preds = []
    all_probs = []
    all_entropies = []
    all_msps = []
    all_mismatches_entropy = []
    all_mismatches_msp = []
    
    for data in tqdm(raw_datasets['test']):
        label = data['pred']
        preds, probs, entropies, msps = generate_with_uncertainty(
            model,
            tokenizer,
            device,
            data['text'],
            max_tokens=max_tokens,
        )
        mismatch = compute_mismatch(label, preds, entropies, msps)


        all_labels.append(label)
        all_preds.append(preds)
        all_probs.append(probs)
        all_entropies.append(entropies)
        all_msps.append(msps)
        all_mismatches_entropy.append(mismatch['entropy'])
        all_mismatches_msp.append(mismatch['msp'])

    metrics_report = {}
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

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_dir", default="/nas/user10_1/Counsell/log/Qwen2.5/250413_Qwen/Qwen2.5-3B_base_ep(20)_lr(1e-5)_batch(2)_alpha(0.1)/checkpoint-1280", type=str)
    parser.add_argument("--save_dir", default="/home/user10/vscode/counsell/counsell_new/MLC/model/LLM/QWEN2.5-3B_1e-5/ckpt-1280", type=str)
    # parser.add_argument("--ckpt_dir", default="/nas/counsell/log/polyglot-5.8b/250413_lr(1e-5)_batch(4)_alpha(0.1)/checkpoint-5120", type=str)
    # parser.add_argument("--save_dir", default="/home/user10/vscode/counsell/counsell_new/MLC/uncertainty/output/polyglot-ko-5.8b/ckpt-5120", type=str)
    parser.add_argument("--test_data", default="./data/processed/finetuning/test.json", type=str)
    parser.add_argument("--max_tokens", type=int, default=7, help="Qwen: 7, polyglot: 4")
    return parser.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    run_test(args.ckpt_dir, args.save_dir, args.test_data, args.max_tokens)


if  __name__ == "__main__":
    main()