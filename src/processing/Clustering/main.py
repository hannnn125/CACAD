from pathlib import Path
import os
import json
import csv
from collections import defaultdict
import yaml
import argparse
from utils.preprocess import append_end_marker
from utils.index import embed_and_assign_to_clusters
from utils.emb import QuestionEmbedder
from utils.hdbscan import HDBSCANClustering
from utils.merge import merge_centroid
from sentence_transformers import SentenceTransformer
from utils.extract_topQ import extract_topQ

def generate_cluster_definitions():
    cluster_definitions = defaultdict(dict)
    with open(os.path.join("prompts", "cluster_details", "unique_cluster_details.csv"), "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            cluster_id = int(row["클러스터번호"])
            if cluster_id < 0:
                continue
            cluster_definitions[row["항목"]][cluster_id] = ""
    with open(os.path.join("prompts", "cluster_details", "cluster_definitions.json"), "w", encoding="utf-8") as f:
        json.dump(cluster_definitions, f, ensure_ascii=False, indent=2)
    return cluster_definitions

def parse_args():
    args = argparse.ArgumentParser()
    args.add_argument("--config_path", type=Path, default="./configs/base_config.yaml")
    args.add_argument("--data_dir", type=Path, default="./data/processed/labeled_dataset")
    args.add_argument("--top_n", type=int, default=10)
    args.add_argument("--seed", type=int, default=42)
    return args.parse_args()

def main():
    args = parse_args()
    config_path = args.config_path
    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    data_dir = args.data_dir
    cache_dir = config["paths"]["cache_dir"]
    model_name = config["clustering"]["model_name"]
    abuse_types = config["abuse_types"]
    hdbscan_parameters = {
        k: v
        for k, v in config.get("clustering", {}).get("hdbscan_parameters", {}).items()
        if k in abuse_types
    }
    if not hdbscan_parameters:
        raise ValueError(f"No valid abuse types found in config: {abuse_types}")
    
    model = SentenceTransformer(model_name, cache_folder=cache_dir)

    question_emb = QuestionEmbedder(data_dir, model)
    question_emb.load_questions()
    question_emb.generate_embeddings()

    hdbscan_clustering = HDBSCANClustering(data_dir, hdbscan_parameters)
    centroids_data = hdbscan_clustering.run()

    embed_and_assign_to_clusters(data_dir, centroids_data, model, "val", config["clustering"]["similarity_threshold"] )
    embed_and_assign_to_clusters(data_dir, centroids_data, model, "test", config["clustering"]["similarity_threshold"] )

    merge_centroid(data_dir, config["clustering"]["merge_threshold"])

    append_end_marker(data_dir, ["train", "val", "test"])

    extract_topQ(data_dir,args.top_n)

    generate_cluster_definitions()

if __name__ == "__main__":
    main()