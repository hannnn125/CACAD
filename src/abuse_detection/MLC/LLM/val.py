import logging
import math
import os
import sys
import argparse
from dataclasses import dataclass, field
from itertools import chain
from typing import Optional
from tqdm import tqdm
import re
import datasets
import torch
from datasets import load_dataset

# import tensor_parallel as tp
import transformers
from transformers import (
    # CONFIG_MAPPING,
    # MODEL_FOR_CAUSAL_LM_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    # HfArgumentParser,
    # Trainer,
    # TrainingArguments,
    # default_data_collator,
    # is_torch_tpu_available,
    # set_seed,
)
# from transformers.testing_utils import CaptureLogger
# from transformers.trainer_utils import get_last_checkpoint
# from transformers.utils import check_min_version, send_example_telemetry
# from transformers.utils.versions import require_version

import numpy as np
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
    confusion_matrix,
    classification_report
)

def get_checkpoint_dirs(base_dir):
    ckpt_dirs = []
    for name in os.listdir(base_dir):
        if re.match(r"checkpoint-\d+", name):
            path = os.path.join(base_dir, name)
            if os.path.isdir(path):
                ckpt_dirs.append(path)
    ckpt_dirs.sort(key=lambda p: int(os.path.basename(p).split("-")[1]))
    return ckpt_dirs

def gen(model, tokenizer, device, text, max_new_tokens=7):
    prompt = text + "<|pred|>"
    inputs = tokenizer(prompt, return_tensors='pt', return_token_type_ids=False).to(device)

    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
    )
    
    input_length = inputs['input_ids'].shape[1]
    gen_ids = output_ids[0][input_length:]
    generated_part = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

    tokens = generated_part.split()
    if len(tokens) > max_new_tokens:
        tokens = tokens[:max_new_tokens]

    try:
        pred_list = [int(token) for token in tokens]
    except ValueError:
        pred_list = "error"

    return pred_list, generated_part


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def run_val(checkpoint_path, save_dir, val_data, max_new_tokens=7):
    print(f"\n▶ Running validation for: {checkpoint_path}")
    print(f"  max_new_tokens: {max_new_tokens}", flush=True)

    device = get_device()
    if device.type == "cuda":
        print(f"  device: cuda ({torch.cuda.get_device_name(0)})", flush=True)
    else:
        print(f"  device: cpu", flush=True)

    raw_datasets = load_dataset("json", data_files={"validation": val_data})
    
    config = AutoConfig.from_pretrained(checkpoint_path)
    model = AutoModelForCausalLM.from_pretrained(checkpoint_path, config=config)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)

    if len(tokenizer) > model.get_input_embeddings().weight.shape[0]:
        model.resize_token_embeddings(len(tokenizer))

    model.to(device)
    model.eval()

    os.makedirs(save_dir, exist_ok=True)
    log_file = os.path.join(save_dir, "val_results.csv")
    f = open(log_file, 'w', encoding='utf-8-sig')
    f.write("idx,input_text,ground_truth,predicted,raw_generated\n")

    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    for i, data in enumerate(tqdm(raw_datasets['validation'])):
        input_text = data['text']
        label = data['pred']

        predicted, raw_generated = gen(model, tokenizer, device, input_text, max_new_tokens)
        text_clean = input_text.replace(",", ".").replace("\n", " ")
        f.write(f"{i},{text_clean},{str(label)},{str(predicted)},{raw_generated}\n")

        if label == predicted:
            correct += 1
        total += 1
        all_preds.append(predicted)
        all_labels.append(label)

    f.close()

    accuracy = correct / total if total > 0 else 0.0
    macro_f1 = f1_score(np.array(all_labels), np.array(all_preds), average='macro')

    print(f"Checkpoint {checkpoint_path} 평가 완료")
    print(f"  정확도: {accuracy:.4f} | Macro F1: {macro_f1:.4f}")
    print(f"  예측: {log_file}")

    step = os.path.basename(checkpoint_path).split("-")[1]
    return {
        "step": step,
        "checkpoint": checkpoint_path,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
    }


def write_summary(summary_path, rows):
    with open(summary_path, 'w', encoding='utf-8-sig') as f:
        f.write("step,checkpoint,accuracy,macro_f1\n")
        for row in rows:
            f.write(
                f"{row['step']},{row['checkpoint']},{row['accuracy']},{row['macro_f1']}\n"
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_ckpt_dir", default="/nas/counsell/log/test/", type=str)
    parser.add_argument("--base_save_dir", default="./src/detection/MLC/LLM/output/val", type=str)
    parser.add_argument("--val_data", default="./data/processed/finetuning/val.json", type=str)
    parser.add_argument("--max_new_tokens", type=int, default=7, help="max new tokens for generation")
    args = parser.parse_args()
    os.makedirs(args.base_save_dir, exist_ok=True)
    summary_path = os.path.join(args.base_save_dir, "summary.csv")
    summary_rows = []

    for ckpt_path in get_checkpoint_dirs(args.base_ckpt_dir):
        step = os.path.basename(ckpt_path).split("-")[1]
        save_path = os.path.join(args.base_save_dir, f"ckpt-{step}")
        result = run_val(
            ckpt_path,
            save_path,
            args.val_data,
            max_new_tokens=args.max_new_tokens,
        )
        summary_rows.append(result)
        write_summary(summary_path, summary_rows)

