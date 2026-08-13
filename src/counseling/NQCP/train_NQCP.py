import argparse
import yaml
import os
from pathlib import Path
import json
import pandas as pd
import torch
from dataloader.dataset import NQCPDataset, MultiTurnNQCPDataset
from transformers import DataCollatorWithPadding
from utils.metric import compute_metrics
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--abuse_type', type=str, required=True, choices=['방임', '정서학대', '신체학대', '성학대'])
    parser.add_argument('--model_name', type=str, default="roberta-large")
    parser.add_argument('--lr', type=float, default=2e-5, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size per device')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    return parser.parse_args()

def main():
    args = parse_args()
    abuse_type = args.abuse_type
    model_name = args.model_name
    model_folder_name = model_name.replace('/', '_')
    project_path = Path(__file__).resolve().parents[3]
    config_path = project_path / "configs" / "base_config.yaml"
    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    base_dir = config['paths']["labeled_data"]
    cache_dir = config['paths']["cache_dir"]
    run_name = f"ep{args.epochs}_lr{args.lr}_bs{args.batch_size}"
    run_dir = os.path.join(
        config['paths']["output_dir"], "NQCP", model_folder_name, run_name, abuse_type
    )
    os.makedirs(run_dir, exist_ok=True)

    print('데이터 로딩 중...')
    train_folder = os.path.join(base_dir, "train")
    test_folder = os.path.join(base_dir, "test")
    val_folder = os.path.join(base_dir, "val")    


    train_ds = MultiTurnNQCPDataset(
        folder_path=train_folder,
        tokenizer_name=model_name,
        target_abuse_type=abuse_type
    )

    val_ds = MultiTurnNQCPDataset(
        folder_path=val_folder,
        tokenizer_name=model_name,
        target_abuse_type=abuse_type,
        label2id=train_ds.label2id
    )

    test_ds = MultiTurnNQCPDataset(
        folder_path=test_folder,
        tokenizer_name=model_name,
        target_abuse_type=abuse_type,
        label2id=train_ds.label2id
    )

    # train_ds = NQCPDataset(
    # folder_path=train_folder, 
    # tokenizer_name=model_name,
    # target_abuse_type=abuse_type)

    # val_ds = NQCPDataset(
    # folder_path=val_folder, 
    # tokenizer_name=model_name,
    # target_abuse_type=abuse_type,
    # label2id=train_ds.label2id)

    # test_ds = NQCPDataset(
    # folder_path=test_folder, 
    # tokenizer_name=model_name,
    # target_abuse_type=abuse_type,
    # label2id=train_ds.label2id)

    data_collator = DataCollatorWithPadding(
    tokenizer=train_ds.tokenizer,
    padding=True)

    print(f'▶▶▶ 학습 정보: {abuse_type}')
    print(f'- Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}')
    print(f'- Num Labels (Clusters): {len(train_ds.label2id)}')

    # Model 초기화
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(train_ds.label2id),
        cache_dir=cache_dir)

    # Training Arguments 설정
    training_args = TrainingArguments(
        output_dir=os.path.join(run_dir, "checkpoints"),
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=args.weight_decay,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        logging_dir=os.path.join(run_dir, "logs"),
        logging_steps=100,
        fp16=torch.cuda.is_available(), 
        seed=args.seed, 
    )

     # Trainer 초기화
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
        data_collator=data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )   

    #학습 진행
    print(f'{abuse_type} 모델 학습 시작...')
    trainer.train()

    # 최종 모델 및 매핑 정보 저장
    print(f'학습 완료. 모델 저장 중: {run_dir}')
    trainer.save_model(run_dir)
    train_ds.tokenizer.save_pretrained(run_dir)

    with open(os.path.join(run_dir, "label2id.json"), "w", encoding="utf-8") as f:
        json.dump(train_ds.label2id, f, ensure_ascii=False, indent=4)

    print(f'{abuse_type} 테스트 셋 평가 결과:')
    test_results = trainer.evaluate(test_ds)
    print(test_results)

    new_result_data = {
        "Model": model_name,
        "Abuse_Type": abuse_type,
        "Epochs": args.epochs,
        "Batch_Size": args.batch_size,
        "LR": args.lr,
        "Accuracy": round(test_results["eval_accuracy"], 4),
        "Macro_F1": round(test_results["eval_macro_f1"], 4)
    }

    csv_file_path = os.path.join(
        config['paths']["output_dir"], "NQCP", model_name, f"{run_name}.csv"
    )
    if os.path.exists(csv_file_path):
        df = pd.read_csv(csv_file_path)
        df = pd.concat([df, pd.DataFrame([new_result_data])], ignore_index=True)
    else:
        df = pd.DataFrame([new_result_data])

    df.to_csv(csv_file_path, index=False, encoding='utf-8-sig')
    print(f'CSV result updated: {csv_file_path}')

if __name__ == "__main__":
    main()
