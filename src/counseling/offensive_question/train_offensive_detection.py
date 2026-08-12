import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import argparse
import torch
import numpy as np
import pandas as pd
import json
import time
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments
)
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, classification_report
from datetime import datetime
from tqdm import tqdm


class OffensiveDataset(Dataset):
    """Offensive Question Detection Dataset"""
    def __init__(self, dataframe, tokenizer, max_length):
        self.data = dataframe.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        text = str(row['text']) if pd.notna(row['text']) else ""
        label = int(row['label'])

        inputs = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )

        item = {key: val.squeeze() for key, val in inputs.items()}
        item['labels'] = torch.tensor(label, dtype=torch.long)
        return item


def compute_metrics(eval_pred):
    """평가 메트릭 계산"""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=-1)
    
    accuracy = accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions, average='binary', zero_division=0)
    recall = recall_score(labels, predictions, average='binary', zero_division=0)
    f1 = f1_score(labels, predictions, average='binary', zero_division=0)
    
    return {
        "eval_accuracy": accuracy,
        "eval_precision": precision,
        "eval_recall": recall,
        "eval_f1": f1
    }


def collate_fn(features):
    """배치 collate 함수"""
    return {
        key: torch.stack([f[key] for f in features])
        for key in features[0]
    }


def measure_inference_time(model, test_dataset, device, batch_size=1):
    """Test 데이터셋에 대한 inference 시간 측정"""
    model.eval()
    model.to(device)
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )
    
    latencies = []
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Inference", disable=True):
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch.pop('labels')
            
            start_time = time.time()
            outputs = model(**batch)
            end_time = time.time()
            
            latency = (end_time - start_time) * 1000
            latencies.append(latency)
            
            predictions = torch.argmax(outputs.logits, dim=-1)
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    latencies = np.array(latencies)
    mean_latency = np.mean(latencies)
    std_latency = np.std(latencies)
    min_latency = np.min(latencies)
    max_latency = np.max(latencies)
    median_latency = np.median(latencies)
    
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions, average='binary', zero_division=0)
    recall = recall_score(all_labels, all_predictions, average='binary', zero_division=0)
    f1 = f1_score(all_labels, all_predictions, average='binary', zero_division=0)
    
    inference_stats = {
        "batch_size": batch_size,
        "total_samples": len(test_dataset),
        "mean_latency_ms": float(mean_latency),
        "std_latency_ms": float(std_latency),
        "min_latency_ms": float(min_latency),
        "max_latency_ms": float(max_latency),
        "median_latency_ms": float(median_latency),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1)
    }
    
    return inference_stats, latencies


def train_single_model(model_name, args, train_df, valid_df, test_df):
    """단일 모델 학습 및 평가"""
    print(f"\n[{model_name}] Starting training...")
    
    cache_dir = args.cache_dir
    os.makedirs(cache_dir, exist_ok=True)
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_folder_name = model_name.replace('/', '_')
    run_dir = os.path.join(args.output_dir, "offensive", f'{timestamp}_{model_folder_name}')
    os.makedirs(run_dir, exist_ok=True)

    train_dataset = OffensiveDataset(train_df, tokenizer, args.max_length)
    valid_dataset = OffensiveDataset(valid_df, tokenizer, args.max_length)
    test_dataset = OffensiveDataset(test_df, tokenizer, args.max_length)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,
        cache_dir=cache_dir
    )

    # 학습 설정
    training_args = TrainingArguments(
        output_dir=run_dir,
        num_train_epochs=args.epoch,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        weight_decay=args.weight_decay,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        metric_for_best_model="eval_accuracy",
        greater_is_better=True,
        save_total_limit=1,
        load_best_model_at_end=True,
        report_to=[],
        logging_dir=f'{run_dir}/logs',
        seed=42,
        dataloader_num_workers=4,
        remove_unused_columns=False,
        fp16=True,
        optim="adafactor",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        data_collator=collate_fn
    )

    train_start_time = time.time()
    trainer.train()
    train_end_time = time.time()
    training_time = train_end_time - train_start_time

    test_results = trainer.evaluate(eval_dataset=test_dataset)
    test_acc = test_results['eval_accuracy']
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inference_stats, latencies = measure_inference_time(
        trainer.model, 
        test_dataset, 
        device, 
        batch_size=1
    )
    
    inference_result_path = os.path.join(run_dir, "inference_stats.json")
    inference_results = {
        "inference_stats": inference_stats,
        "test_accuracy": float(test_acc)
    }
    with open(inference_result_path, "w", encoding="utf-8") as f:
        json.dump(inference_results, f, ensure_ascii=False, indent=2)
    
    config_path = os.path.join(run_dir, "training_config.json")
    config = {
        "model_name": model_name,
        "epochs": args.epoch,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "weight_decay": args.weight_decay,
        "max_length": args.max_length,
        "data_dir": args.data_dir,
        "train_samples": len(train_df),
        "valid_samples": len(valid_df),
        "test_samples": len(test_df),
        "training_time_seconds": training_time,
        "inference_mean_latency_ms": inference_stats['mean_latency_ms'],
        "test_accuracy": float(test_acc),
        "timestamp": timestamp
    }
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    
    print(f"[{model_name}] Training completed - Test Acc: {test_acc:.4f}, Inference: {inference_stats['mean_latency_ms']:.2f}ms")
    
    return {
        "model_name": model_name,
        "run_dir": run_dir,
        "test_accuracy": test_acc,
        "inference_stats": inference_results,
        "training_time": training_time
    }


def main():
    parser = argparse.ArgumentParser(description="Offensive Question Detection Training (Multi-Model)")
    parser.add_argument('--models', type=str, nargs='+', 
                        default=['klue/bert-base', 'klue/roberta-large', 'BM-K/KoSimCSE-roberta'],
                        help="List of model names to train")
    parser.add_argument('--epoch', type=int, default=15, help="Number of epochs")
    parser.add_argument('--batch_size', type=int, default=16, help="Batch size")
    parser.add_argument('--lr', type=float, default=1e-5, help="Learning rate")
    parser.add_argument('--weight_decay', type=float, default=1e-2, help="Weight decay")
    parser.add_argument('--max_length', type=int, default=512, help="Max sequence length")
    parser.add_argument('--data_dir', type=str, default='/home/user10/vscode/counsell/offensive_dataset', 
                        help="Data directory")
    parser.add_argument('--output_dir', type=str, default='/nas/user10_a6000/offensive_runs',
                        help="Output directory")
    parser.add_argument('--cache_dir', type=str, default='/nas/user10/.cache/yg/',
                        help="Cache directory")
    args = parser.parse_args()

    print(f"\nTraining {len(args.models)} models with {args.epoch} epochs")

    train_path = os.path.join(args.data_dir, "train.csv")
    valid_path = os.path.join(args.data_dir, "valid.csv")
    test_path = os.path.join(args.data_dir, "test.csv")
    
    train_df = pd.read_csv(
        train_path, 
        encoding='utf-8-sig', 
        quoting=1, 
        escapechar='\\', 
        doublequote=True
    )
    valid_df = pd.read_csv(
        valid_path,
        encoding='utf-8-sig',
        quoting=1,
        escapechar='\\',
        doublequote=True
    )
    test_df = pd.read_csv(
        test_path,
        encoding='utf-8-sig',
        quoting=1,
        escapechar='\\',
        doublequote=True
    )
    
    all_results = []
    for model_name in args.models:
        try:
            result = train_single_model(model_name, args, train_df, valid_df, test_df)
            all_results.append(result)
        except Exception as e:
            print(f"Error training {model_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print("\n" + "="*80)
    print("Summary")
    print("="*80)
    
    summary_data = []
    for result in all_results:
        summary_data.append({
            "Model": result["model_name"],
            "Test Accuracy": f"{result['test_accuracy']:.4f}",
            "Inference Latency (ms)": f"{result['inference_stats']['inference_stats']['mean_latency_ms']:.2f}",
            "Training Time (s)": f"{result['training_time']:.2f}"
        })
    
    summary_df = pd.DataFrame(summary_data)
    print(summary_df.to_string(index=False))
    
    summary_path = os.path.join(args.output_dir, f"summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"\nSummary saved: {summary_path}")
    print("Completed\n")


if __name__ == "__main__":
    main()
