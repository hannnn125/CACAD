import os 
import json 
import yaml
import argparse
import random
import ast
from pathlib import Path
from typing import TextIO
from collections import defaultdict

def dialogue_from_data(data) -> str:
    dialogue = {}
    for item in data["list"]:
        category = item["항목"]
        audio = "\n".join(
            [f"Q: {x['text']}" if x["type"] == "Q" else f"A: {x['text']}" for x in item["audio"]]
        )
        dialogue[category] = audio
    return dialogue

def sample_files(args, sample_num):
    "ground_truth가 다른 n 개의 파일을 샘플링"
    train_dir = os.path.join(args.input_dir,"train")
    with open(os.path.join(args.input_dir,"split_distribution.json"), "r", encoding="utf-8") as f:
        distribution=json.load(f)
    label_counts = distribution["train"]['combo_counts']
    rank = sorted(label_counts.keys(), key=label_counts.get, reverse=True)
    top_labels = rank[:sample_num]    
    top_keys = [tuple(ast.literal_eval(s))for s in top_labels]
    by_label = {key:[] for key in top_keys}
    for file in os.listdir(train_dir):
        with open(os.path.join(train_dir,file), "r", encoding="utf-8") as f:
            data = json.load(f)
        key = tuple(data["ground_truth"])
        if key in by_label:
            by_label[key].append((file,data))
    rng = random.Random(args.seed)
    samples = []
    for key in top_keys:
        candidates = sorted(by_label[key], key=lambda x: x[0])
        fname, data = rng.choice(candidates)
        samples.append({"filename": fname, "data": data})
    return samples

def gen_instruction_prompt(args, example_num):
    samples = sample_files(args,example_num)
    exclude_files = [sample["filename"] for sample in samples]
    instruction_prompt = """
        너는 아동학대판별상담사로써 각 항목별 대화를 바탕으로 실제 학대 여부를 판단해야해. 다음은 아동과의 상담 대화야. 
    대화는 총 4개의 항목(방임, 정서학대, 신체학대, 성학대)으로 구분되어 있고, 각 항목은 <|항목명|> 형태의 태그로 시작한 후 해당 항목에 대한 Q&A 형식의 대화가 이어져. 
    <|pred|> 뒤는 각 항목의 학대여부를 나타내는 라벨이야 다음과 같은 형태로 [방임여부, 정서학대여부, 신체학대여부, 성학대여부] 학대에 해당하면 1, 해당하지 않으면 0으로 표시돼.
        """
    for i in range(example_num):
        dialogue = dialogue_from_data(samples[i]["data"])
        example = "".join([f"<|{category}|>\n{audio}\n" for category, audio in dialogue.items()]) + f"<|pred|> {samples[i]['data']['ground_truth']}\n"
        instruction_prompt += f"\n## 예시 {i+1}\n{example}\n" 
    return exclude_files, instruction_prompt + "이제 주어진 대화를 보고 학대여부를 판별하면 돼."
 
def generate_ft_dataset(data_path_list: list, instruction_template: str, out_file: TextIO):
    for data_path in data_path_list:
        with open(data_path, encoding="utf-8") as f:
            data = json.load(f)
        dialogue = ""
        for item in data["list"]:
            category = item["항목"]
            audio = "\n".join(
                [f"Q: {x['text']}" if x["type"] == "Q" else f"A: {x['text']}" for x in item["audio"]]
            )
            dialogue += f"<|{category}|>\n{audio}\n"

        sample_obj = {
            "text": (instruction_template + "\n" + dialogue).strip(),
            "pred": data["ground_truth"],
        }
        out_file.write(json.dumps(sample_obj, ensure_ascii=False) + "\n")

def parse_args():
    args = argparse.ArgumentParser()
    args.add_argument("--config_path", type=str, default="configs/base_config.yaml")
    args.add_argument("--example_num", type=int, default=2)
    args.add_argument("--abuse_types", type=list, default=["방임", "정서학대", "신체학대", "성학대"])
    args.add_argument("--input_dir", type=str, default="data/processed/labeled_dataset")
    args.add_argument("--output_dir", type=str, default="data/processed/finetuning_dataset")
    args.add_argument("--seed", type=int, default=42)
    return args.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    exclude_files, instruction_prompt = gen_instruction_prompt(args, example_num = args.example_num)
    print(exclude_files)
    config = yaml.load(open(args.config_path, "r", encoding="utf-8"), Loader=yaml.FullLoader)
    config["exclude_files"] = exclude_files
    with open(args.config_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, ensure_ascii=False)

    out_files = {
        split: open(os.path.join(args.output_dir, f"{split}.json"), "w", encoding="utf-8")
        for split in ["train", "val", "test"]
    }

    for split in ["train", "val", "test"]:
        data_dir = os.path.join(args.input_dir, split)
        data_path_list = sorted(
            os.path.join(data_dir, f)
            for f in os.listdir(data_dir)
            if f.endswith(".json") and f not in exclude_files
        )
        generate_ft_dataset(data_path_list, instruction_prompt, out_files[split])

    for f in out_files.values():
        f.close()

if __name__ == "__main__":
    main()