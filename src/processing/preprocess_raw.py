from pathlib import Path
import yaml
import json
import re
import shutil
import numpy as np
from skmultilearn.model_selection import iterative_train_test_split

class dataProcessor: 
    def __init__(self):
        self.label_names = ["방임", "정서학대", "신체학대", "성학대"]
        self.abuse_thresholds = {
                                "방임": 4,
                                "정서학대": 5,
                                "신체학대": 5,
                                "성학대": 5,
                                }   
        self.skipped_files = []

    def process_directory(self,input_dir: Path, output_dir: Path) -> list:
        json_files = sorted(input_dir.glob("*.json"))
        for json_file in json_files:
            try:
                with open(json_file, encoding="utf-8") as f:
                    data = json.load(f)

                result = self.process_abuse_items(data)
                if result is None:
                    self.skipped_files.append(json_file.name)
                    print(f"[SKIP] {json_file.name}: 학대여부 데이터 불완전")
                    continue
                    
                out_file = output_dir / json_file.name
                with open(out_file, "w", encoding="utf-8") as f:
                    json.dump(result, f, ensure_ascii=False, indent=2)

            except Exception as e:
                self.skipped_files.append(json_file.name)
                print(f"Error processing {json_file.name}: {e}")
                continue

        print(f"\n완료: {len(json_files) - len(self.skipped_files)}개 저장, {len(self.skipped_files)}개 스킵")

    def process_abuse_items(self, raw_data: dict) -> dict:
        abuse_section = next((item for item in raw_data.get("list", []) if item.get("문항") == "학대여부"),None)
        if not abuse_section:
            return None
        abuse_items = []
        for entry in abuse_section.get("list", []):
            category = entry.get("항목", "")
            score = entry.get("점수")
            comment = entry.get("임상가코멘트", {}) 
            audio = entry.get("audio", [])

            if score is None or not category or not audio or not comment:
                return None

            comment = comment.get("기타", comment.get("val", "")) \
                    if "기타" in comment.get("val", "") \
                    else comment.get("val", "")
            comment = self.fix_comment(comment)

            item_dict = {
                "항목": self.clean_value(category),
                "점수": self.clean_value(score),
                "임상가코멘트": self.clean_value(comment),
                "label": self.relabel_abuse(category, score),
                "audio": [
                        {"type": self.clean_value(a.get("type", "")), "text": self.clean_value(a.get("text", ""))}
                        for a in audio
                    ],
            }
            abuse_items.append(item_dict)
        label_map = {item["항목"]: item["label"] for item in abuse_items}
        ground_truth = [
            label_map["방임"],
            label_map["정서학대"],
            label_map["신체학대"],
            label_map["성학대"],
        ]
        return {"list": abuse_items, "ground_truth": ground_truth}

    def relabel_abuse(self, category: str, score: int) -> list:
        new_label = []
        if category in self.label_names:
            new_label=int(score >= self.abuse_thresholds[category])
        return new_label

    def fix_comment(self, comment: str) -> str:
        match = re.search(r"\[(.*?)\]", comment)
        if not match:
            raise ValueError(f"Comment pattern not found")
        word = match.group(1)

        if re.fullmatch(r"\d+점", word):
            return comment
        if re.fullmatch(r"\d+", word):
            comment = comment.replace(f"[{word}]", f"[{word}점]")
        else:
            raise ValueError(f"Invalid comment format")
        return comment

    def clean_value(self, value):
        match value: 
            case str():
                return value.replace("||", "").strip()
            case list():
                return [self.clean_value(v) for v in value]
            case dict():
                if set(value.keys()) == {"val"}:
                    return self.clean_value(value["val"])
                else:
                    return {k: self.clean_value(v) for k, v in value.items()}
            case _:
                return value
    
    def stratified_train_test_split(self, data: list):

        X = np.arange(len(data)).reshape(-1, 1)
        y = np.array([item["ground_truth"] for item in data])

        X_train, y_train, X_temp, y_temp = iterative_train_test_split(
            X, y, test_size=0.2
        )

        X_val, y_val, X_test, y_test = iterative_train_test_split(
            X_temp, y_temp, test_size=0.5
        )

        train_data = [data[i] for i in X_train.flatten()]
        val_data   = [data[i] for i in X_val.flatten()]
        test_data  = [data[i] for i in X_test.flatten()]

        return train_data, val_data, test_data


def main():
    project_path = Path(__file__).resolve().parents[2]
    config_path = project_path / "configs" / "base_config.yaml"
    with open(config_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        input_dir = project_path / config["paths"]["base_data"] / "raw"
        output_dir = project_path / config["paths"]["base_data"] / "processed" / "labeled_dataset"
        output_dir.mkdir(parents=True, exist_ok=True)

    processor = dataProcessor()
    processor.process_directory(input_dir, output_dir)
    
    all_data = []
    for json_file in sorted(output_dir.glob("*.json")):
        with open(json_file, encoding="utf-8") as f:
            item = json.load(f)
            item["_filename"] = json_file.name
            all_data.append(item)

    train_data, val_data, test_data = processor.stratified_train_test_split(all_data)

    split_root = output_dir
    for split_name, split_items in [("train", train_data), ("val", val_data), ("test", test_data)]:
        split_dir = split_root / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        for item in split_items:
            filename = item.pop("_filename")
            shutil.move(str(output_dir / filename), str(split_dir / filename))

    print(f"split 완료 → train: {len(train_data)}, val: {len(val_data)}, test: {len(test_data)}")

    label_names = ["방임", "정서학대", "신체학대", "성학대"]

    def split_distribution(split_items):
        from collections import Counter
        label_counts = {name: {"0": 0, "1": 0} for name in label_names}
        combo_counts = Counter(str(item["ground_truth"]) for item in split_items)
        for item in split_items:
            for name, val in zip(label_names, item["ground_truth"]):
                label_counts[name][str(val)] += 1
        return {"total": len(split_items), "per_label": label_counts, "combo_counts": dict(combo_counts)}

    distribution = {
        "train": split_distribution(train_data),
        "val":   split_distribution(val_data),
        "test":  split_distribution(test_data),
    }

    dist_file = output_dir / "split_distribution.json"
    with open(dist_file, "w", encoding="utf-8") as f:
        json.dump(distribution, f, ensure_ascii=False, indent=2)

    print(f"분포 저장 → {dist_file}")

if __name__ == "__main__":
    main()
