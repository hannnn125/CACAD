import os
from pathlib import Path
import json
import numpy as np
from collections import OrderedDict
from sentence_transformers import SentenceTransformer

class QuestionEmbedder: 
    def __init__(self, data_dir: Path, model: SentenceTransformer):
        self.data_dir = data_dir
        self.model = model 
        self.questions_by_type = {}
        self.index_map = {}
        os.makedirs(data_dir / "embeddings", exist_ok=True)
        self.questions_path = data_dir/ "embeddings" /"questions_by_type.json"
        self.index_path = data_dir / "embeddings" /"index_map.json"

    def load_questions(self):
        if self.questions_path.exists() and self.index_path.exists():
            with open(self.questions_path, encoding="utf-8") as f:
                self.questions_by_type = json.load(f)
            with open(self.index_path, encoding="utf-8") as f:    
                self.index_map = json.load(f)
        else:
            self.extract_questions()

    def extract_questions(self):
        for file_name in os.listdir(self.data_dir / "train"):
            with open(self.data_dir / "train" / file_name, encoding="utf-8") as f:
                data = json.load(f)

            for item_idx, item in enumerate(data.get("list", [])):
                abuse_type = item.get("항목")

                for audio_idx, audio in enumerate(item.get("audio", [])):
                    if audio.get("type") == "Q":
                        self.questions_by_type.setdefault(abuse_type, []).append(audio["text"])
                        self.index_map.setdefault(abuse_type, []).append((file_name, item_idx, audio_idx))

        with open(self.questions_path, "w", encoding="utf-8") as f:
            json.dump(self.questions_by_type, f, ensure_ascii=False, indent=2)
        with open(self.index_path, "w", encoding="utf-8") as f:
            json.dump(self.index_map, f, ensure_ascii=False, indent=2)
    
    def generate_embeddings(self):
        emb_dir =self.data_dir/"embeddings"

        for abuse_type, questions in self.questions_by_type.items():
            unique_questions = list(OrderedDict.fromkeys(questions))
            question_to_uid = {q: i for i, q in enumerate(unique_questions)}
            original_to_uid = [question_to_uid[q] for q in questions]

            embeddings = self.model.encode(
                unique_questions,
                batch_size=32,
                show_progress_bar=True,
                normalize_embeddings=True)

            np.save(os.path.join(emb_dir, f"{abuse_type}_unique.npy"), embeddings)

            mapping = {
                "unique_questions": unique_questions,
                "original_to_unique": original_to_uid,
                "index_map": self.index_map.get(abuse_type, []),
            }
            with open(os.path.join(emb_dir, f"{abuse_type}_mapping.json"), "w", encoding="utf-8") as f:
                json.dump(mapping, f, ensure_ascii=False, indent=2)