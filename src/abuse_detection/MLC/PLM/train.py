import argparse
import os
from pathlib import Path
import yaml
import time
import torch
from sklearn.metrics import accuracy_score, classification_report
from transformers import Trainer, TrainingArguments, AutoTokenizer, set_seed

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

from dataloader.dataset import AbuseMultilabelDataset
from PLM.utils.metric import compute_MultiBERT_metrics

import torch.nn as nn
from transformers import AutoModel

class AbuseClassifier(nn.Module):
    def __init__(self, model_name, num_labels=4, cache_dir=None):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name, cache_dir=cache_dir)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        logits = self.classifier(pooled_output)
        if labels is not None:
            loss_fn = nn.BCEWithLogitsLoss()
            loss = loss_fn(logits, labels)
            return {"loss": loss, "logits": logits}
        return {"logits": logits}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="monologg/kobert")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)

    project_path = Path(__file__).resolve().parents[4]
    config_path = project_path / "configs" / "base_config.yaml"
    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    data_path = config['paths']["labeled_data"]
    save_dir = os.path.join(config['paths']["output_dir"], "MLC", "PLM")
    cache_dir = config['paths']["cache_dir"]
    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)

    timestamp = time.strftime("%Y%m%d")
    model_base = args.model_name.replace("/", "-")
    exp_name = f"{timestamp}_{model_base}_ep{args.epochs}_bs{args.batch_size}_lr{args.learning_rate}"

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name, trust_remote_code=True, cache_dir=cache_dir
    )
    if hasattr(tokenizer, "save_vocabulary"):
        original_save_vocab = tokenizer.save_vocabulary

        def patched_save_vocab(save_directory, filename_prefix=None):
            return original_save_vocab(save_directory)

        tokenizer.save_vocabulary = patched_save_vocab

    train_dataset = AbuseMultilabelDataset(
        os.path.join(data_path, "train"), tokenizer
    )
    val_dataset = AbuseMultilabelDataset(
        os.path.join(data_path, "val"), tokenizer
    )
    test_dataset = AbuseMultilabelDataset(
        os.path.join(data_path, "test"), tokenizer
    )

    print(f"Train: {len(train_dataset)} samples")
    print(f"Validation: {len(val_dataset)} samples")
    print(f"Test: {len(test_dataset)} samples")

    training_args = TrainingArguments(
        output_dir=save_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_dir=os.path.join(save_dir, "logs"),
        logging_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        seed=args.seed
    )

    model = AbuseClassifier(model_name=args.model_name, cache_dir=cache_dir)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_MultiBERT_metrics
    )

    trainer.train()
    print("Training Done!")

    test_result = trainer.predict(test_dataset)
    test_preds = (torch.sigmoid(torch.tensor(test_result.predictions)) > 0.5).int().numpy()
    test_labels = test_result.label_ids.astype(int)

    exact_match_acc = accuracy_score(test_labels, test_preds)
    print(f"\nExact Match Accuracy: {exact_match_acc:.4f}")

    report = classification_report(test_labels, test_preds, target_names=["방임", "정서", "신체", "성"])
    print("\n Test Classification Report:")
    print(report)

    with open(os.path.join(save_dir, "test_report.txt"), "w", encoding="utf-8") as f:
        f.write(f"Exact Match Accuracy: {exact_match_acc:.4f}\n\n")
        f.write(report)


if __name__ == "__main__":
    main()
