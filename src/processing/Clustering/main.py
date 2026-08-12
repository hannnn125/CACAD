from pathlib import Path
import yaml

from utils.preprocess import append_end_marker
from utils.index import embed_and_assign_to_clusters
from utils.emb import QuestionEmbedder
from utils.hdbscan import HDBSCANClustering
from utils.merge import merge_centroid
from sentence_transformers import SentenceTransformer

def main():

    project_dir = Path(__file__).resolve().parents[3]
    config_path = project_dir / "configs" / "base_config.yaml"
    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    data_dir = project_dir / config["paths"]["labeled_data"]
    model_name = config["clustering"]["model_name"]
    abuse_types = config["abuse_types"]
    hdbscan_parameters = {
        k: v
        for k, v in config.get("clustering", {}).get("hdbscan_parameters", {}).items()
        if k in abuse_types
    }
    if not hdbscan_parameters:
        raise ValueError(f"No valid abuse types found in config: {abuse_types}")
    
    model = SentenceTransformer(model_name)

    question_emb = QuestionEmbedder(data_dir, model)
    question_emb.load_questions()
    question_emb.generate_embeddings()

    hdbscan_clustering = HDBSCANClustering(data_dir, hdbscan_parameters)
    centroids_data = hdbscan_clustering.run()

    embed_and_assign_to_clusters(data_dir, centroids_data, model, "val", config["clustering"]["similarity_threshold"] )
    embed_and_assign_to_clusters(data_dir, centroids_data, model, "test", config["clustering"]["similarity_threshold"] )

    merge_centroid(data_dir, config["clustering"]["merge_threshold"])

    append_end_marker(data_dir, ["train", "val", "test"])

if __name__ == "__main__":
    main()