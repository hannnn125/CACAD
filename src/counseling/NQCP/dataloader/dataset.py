import os
import json
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

class NQCPDataset(Dataset):
    def __init__(
        self,
        folder_path,
        tokenizer_name,
        target_abuse_type=None, 
        max_length=512,
        label2id=None
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length
        
        # 데이터 로드 및 타입별 분류
        all_samples = self._build_samples(folder_path)
        
        # 특정 타입 필터링
        if target_abuse_type:
            self.samples = [s for s in all_samples if s["abuse_type"] == target_abuse_type]
        else:
            self.samples = all_samples

        if not self.samples:
            raise ValueError(f"데이터가 없습니다. (타입: {target_abuse_type})")

        # Label Mapping 
        self.texts = [s["history"] for s in self.samples]
        self.labels_raw = [s["next_cluster"] for s in self.samples]

        if label2id is None:
            # 해당 데이터에 존재하는 클러스터만 추출하여 0번부터 매핑
            unique_labels = sorted(list(set(self.labels_raw)))
            self.label2id = {label: i for i, label in enumerate(unique_labels)}
        else:
            self.label2id = label2id

        self.id2label = {v: k for k, v in self.label2id.items()}
        self.labels = [self.label2id[l] for l in self.labels_raw]

    def _build_samples(self, folder_path):
        samples = []
        for fname in os.listdir(folder_path):
            if not fname.endswith(".json"): continue
            
            with open(os.path.join(folder_path, fname), encoding="utf-8") as f:
                data = json.load(f)

            for section in data.get("list", []):
                abuse_type = section.get("항목")
                audio = section.get("audio", [])
                
                qa = []
                it = iter(audio)
                for item in it:
                    if item.get("type") == "Q" and "cluster" in item:
                        nxt = next(it, None)
                        if nxt and nxt.get("type") == "A":
                            q_text = item["text"].strip()
                            a_text = nxt["text"].strip()
                            qa.append((q_text, a_text, item["cluster"]))

                for i in range(len(qa) - 1):
                    if qa[i+1][2] != -1:
                        samples.append({
                            "abuse_type": abuse_type,
                            "history": f"Q: {qa[i][0]}\nA: {qa[i][1]}",
                            "next_cluster": qa[i + 1][2]
                        })
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.tokenizer(
            self.texts[idx],
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        return {
            "input_ids": item["input_ids"].squeeze(0),
            "attention_mask": item["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long)
        }

class MultiTurnNQCPDataset(Dataset):
    def __init__(
        self,
        folder_path,
        tokenizer_name,
        target_abuse_type=None,
        max_length=512,
        label2id=None,
        max_turns=None  # None이면 토큰 기준 truncation만 사용
    ):
        # tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.tokenizer.truncation_side = "left"  
        self.max_length = max_length
        self.max_turns = max_turns

        # 데이터 로드
        all_samples = self._build_samples(folder_path)

        # abuse type 필터링
        if target_abuse_type:
            self.samples = [s for s in all_samples if s["abuse_type"] == target_abuse_type]
        else:
            self.samples = all_samples

        if not self.samples:
            raise ValueError(f"데이터가 없습니다. (타입: {target_abuse_type})")

        # label mapping
        self.texts = [s["history"] for s in self.samples]
        self.labels_raw = [s["next_cluster"] for s in self.samples]

        if label2id is None:
            unique_labels = sorted(set(self.labels_raw))
            self.label2id = {l: i for i, l in enumerate(unique_labels)}
        else:
            self.label2id = label2id

        self.id2label = {v: k for k, v in self.label2id.items()}
        self.labels = [self.label2id[l] for l in self.labels_raw]

    def _build_samples(self, folder_path):
        samples = []
        for fname in os.listdir(folder_path):
            if not fname.endswith(".json"): continue
            with open(os.path.join(folder_path, fname), encoding="utf-8") as f:
                data = json.load(f)

            for section in data.get("list", []):
                abuse_type = section.get("항목")
                audio = section.get("audio", [])

                qa = []
                it = iter(audio)
                for item in it:
                    if item.get("type") == "Q" and "cluster" in item:
                        cluster_val = item["cluster"]
                        nxt = next(it, None)
                        q_text = item["text"].strip()
                        if nxt and nxt.get("type") == "A":
                            a_text = nxt["text"].strip()
                            qa.append((q_text, a_text, cluster_val))
                        elif cluster_val == -2:
                            # 상담 종료 시점이므로 답변이 없어도 추가
                            qa.append((q_text, "", cluster_val))

                for i in range(len(qa)):
                    if i + 1 >= len(qa): 
                        break 

                    target_cluster = qa[i + 1][2]
                    if target_cluster == -1: # -1은 제외
                        continue

                    history = []
                    start = 0
                    if self.max_turns is not None:
                        start = max(0, i + 1 - self.max_turns)

                    # i번째 대화까지 모두 history에 포함
                    for j in range(start, i + 1):
                        history.append(f"Q: {qa[j][0]}\nA: {qa[j][1]}")

                    samples.append({
                        "abuse_type": abuse_type,
                        "history": "\n".join(history).strip(),
                        "next_cluster": target_cluster # 여기서 -2 라벨이 포함
                    })
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        encoded = self.tokenizer(
            self.texts[idx],
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )

        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long)
        }