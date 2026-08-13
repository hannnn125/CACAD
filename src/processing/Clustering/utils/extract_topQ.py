import os
import json
from collections import defaultdict, Counter

def extract_topQ(data_dir, top_n = 5):
    counts = defaultdict(lambda:defaultdict(Counter))
    train_dir = os.path.join(data_dir,"train")
    for file in os.listdir(train_dir):
        with open(os.path.join(train_dir, file), "r", encoding="utf-8") as f:
            data = json.load(f)
        for item in data["list"]:
            category = item["항목"]
            for audio in item["audio"]:
                if audio["type"] == "Q":
                    text = audio.get("text", "").strip()
                    cluster = audio.get("cluster")

                if cluster < 0 : 
                    continue 
                counts[category][cluster][text] += 1                 
        
    topQ_data = defaultdict(list)
    for category, cluster_map in counts.items():
        for cluster, text_counter in sorted(cluster_map.items()):
            for text, _freq in text_counter.most_common(top_n):
                topQ_data[category].append({
                    "cluster": cluster,
                    "text": text,
                })
    out_path = os.path.join("prompts", "cluster_details", "TopQ_by_type.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(topQ_data, f, ensure_ascii=False, indent=4)
    print(f"Saved TopQ data to {out_path}")