import os
import json
import torch
from torch.utils.data import Dataset


class AbuseMultilabelDataset(Dataset):
    def __init__(self, json_dir, tokenizer, max_len_per_item=128):
        self.data = []
        self.tokenizer = tokenizer
        self.max_len_per_item = max_len_per_item
        self.label_map = {"방임": 0, "정서학대": 1, "신체학대": 2, "성학대": 3}

        for file in os.listdir(json_dir):
            if not file.endswith(".json"):
                continue
            items = json.load(open(os.path.join(json_dir, file), encoding="utf-8"))

            label = items["ground_truth"]
            segments = []

            for item in items.get("list", []):
                category = item.get("항목")
                audio = " ".join(x["text"] for x in item.get("audio", []))
                text = f"[{category}] {audio}"
                
                tokenized = self.tokenizer(
                    text,
                    padding="max_length",
                    truncation=True,
                    max_length=self.max_len_per_item,
                    return_tensors="pt",
                )
                segments.append(tokenized)

            if len(segments) == 0:
                continue

            # 최대 4개 항목, 없으면 빈 segment로 채우기
            while len(segments) < 4:
                segments.append(
                    self.tokenizer(
                        "",
                        padding="max_length",
                        truncation=True,
                        max_length=self.max_len_per_item,
                        return_tensors="pt",
                    )
                )

            merged = {}
            for k in segments[0]:
                merged[k] = torch.cat([seg[k] for seg in segments], dim=1).squeeze()
            merged["labels"] = torch.tensor(label, dtype=torch.float)
            self.data.append(merged)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
