import os
from pathlib import Path
import hdbscan
import numpy as np
import csv
import json
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity

class HDBSCANClustering:
    def __init__(self, data_directory: str, hdbscan_parameters: dict):
        self.data_dir = data_directory
        self.parameters = hdbscan_parameters
        self.emb_dir = os.path.join(self.data_dir, "embeddings")
        
        self.json_cache = {}
        self.cluster_details = []
        self.noise_reassign_stats = defaultdict(lambda: {
            "total_noise": 0,
            "reassigned": 0
        })

        self.centroid_dir = os.path.join(self.emb_dir, "centroids")
        os.makedirs(self.centroid_dir, exist_ok=True)

    def run(self):
        centroids_data = {} #centroids초기화

        for emb_filename in os.listdir(self.emb_dir):
            if not emb_filename.endswith("_unique.npy"):
                continue

            abuse_type = emb_filename.replace("_unique.npy", "")

            embeddings, mapping = self.load_embeddings_and_mapping(abuse_type) #클러스터링 전 데이터 로드
            cluster_ID = self.clustering(embeddings, abuse_type) #클러스터링
            
            centroids = self.calculate_centroids(abuse_type, embeddings, cluster_ID) #centroids 계산 (noise 값 제외)
            centroids_data[abuse_type] = centroids #centroids 저장

            self.reassign_noise_to_clusters(abuse_type, embeddings, cluster_ID, centroids) #noise(-1) 재할당
            self.collect_cluster_details(abuse_type, mapping["unique_questions"], cluster_ID) #매핑을 위한 정보 저장
            
            full_cluster_ID = self.expand_to_original(cluster_ID, mapping["original_to_unique"])
            self.inject_clusters_to_json(abuse_type, mapping, full_cluster_ID) #json 파일에 클러스터 정보 주입
        
        self.save_modified_json()
        # self.save_cluster_details_csv()

        print ("noise reassign stats--------------------------------")
        for abuse_type, stats in self.noise_reassign_stats.items():
            total = stats["total_noise"]
            reassigned = stats["reassigned"]
            ratio = (reassigned / total * 100) if total > 0 else 0
            print(f"[{abuse_type}] Noise: {total} → Reassigned: {reassigned} ({ratio:.1f}%)")
        print ("-----------------------------------------------------")
        
        return centroids_data

    def load_embeddings_and_mapping(self, abuse_type: str):
        embeddings = np.load(os.path.join(self.emb_dir, f"{abuse_type}_unique.npy"))
        with open(os.path.join(self.emb_dir, f"{abuse_type}_mapping.json"),encoding="utf-8") as f:
            mapping = json.load(f) 
        return embeddings, mapping

    def clustering(self, embeddings: np.ndarray, abuse_type: str):
        params = self.parameters[abuse_type]
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=params["min_cluster_size"],
            min_samples=params["min_samples"],
            metric="euclidean" 
        )
        return clusterer.fit_predict(embeddings)

    def calculate_centroids(self, abuse_type: str, embeddings: np.ndarray, cluster_ID: list):
        centroids = {}
        for cluster_id in set(cluster_ID):
            if cluster_id == -1: 
                continue 
            cluster_embeddings = embeddings[cluster_ID == cluster_id]
            centroids[int(cluster_id)] = cluster_embeddings.mean(axis=0)
        np.save(os.path.join(self.centroid_dir, f"{abuse_type}_centroids.npy"), centroids)
        return centroids

    def reassign_noise_to_clusters(
        self, 
        abuse_type: str, 
        embeddings: np.ndarray, 
        cluster_ID: list, 
        centroids: dict,
        similarity_threshold=0.55):

        noise_indices = np.where(cluster_ID == -1)[0]
        if len(noise_indices) == 0 or not centroids:
            return

        self.noise_reassign_stats[abuse_type]["total_noise"] += len(noise_indices)

        noise_embeddings = embeddings[noise_indices]
        cluster_ids = np.array(list(centroids.keys()))
        centroid_matrix = np.vstack([centroids[cid] for cid in cluster_ids])
        similarities = cosine_similarity(noise_embeddings, centroid_matrix)

        max_sims = np.max(similarities, axis=1)
        best_cluster_indices = np.argmax(similarities, axis=1)
        mask = max_sims >= similarity_threshold
        
        reassigned_cluster_ids = cluster_ids[best_cluster_indices[mask]]
        cluster_ID[noise_indices[mask]] = reassigned_cluster_ids

        self.noise_reassign_stats[abuse_type]["reassigned"] += np.sum(mask)
    
    def collect_cluster_details(self, abuse_type: str, unique_questions: list, cluster_ID: list):
        for question, cluster_id in zip(unique_questions, cluster_ID):
            self.cluster_details.append({
                "abuse_type": abuse_type,
                "cluster_id": int(cluster_id),
                "question": question
            })
        return self.cluster_details

    def expand_to_original(self, cluster_ID: list, mapping: dict):
        return [cluster_ID[uid] for uid in mapping]


    def inject_clusters_to_json(self, abuse_type: str, mapping: dict, cluster_ID: list):
        index_map = mapping["index_map"]
        unique_questions = mapping["unique_questions"]
        original_to_unique = mapping["original_to_unique"]

        for (filename, item_idx, audio_idx), cluster_id, uid in zip(
            index_map,
            cluster_ID,
            original_to_unique
        ):
            if filename not in self.json_cache:
                path = os.path.join(self.data_dir, "train", filename)
                with open(path, encoding="utf-8") as f:
                    self.json_cache[filename] = json.load(f)

            self.json_cache[filename]["list"][item_idx]["audio"][audio_idx][
                "cluster"
            ] = int(cluster_id)

    def save_modified_json(self):
        output_path = os.path.join(self.data_dir, "train")
        os.makedirs(output_path, exist_ok=True)

        for filename, data in self.json_cache.items():
            with open(
                os.path.join(output_path, filename),
                "w",
                encoding="utf-8"
            ) as f:
                json.dump(data, f, ensure_ascii=False, indent=2)


    def save_cluster_details_csv(self):
        selected_types = set(self.parameters.keys())
        self.cluster_details.sort(
            key=lambda x: (x["abuse_type"], x["cluster_id"])
        )

        out_dir = os.path.join(
            self.data_dir,
            "cluster_details"
        )
        os.makedirs(out_dir, exist_ok=True)

        csv_path = os.path.join(out_dir, "unique_cluster_details.csv")

        existing_rows = []
        if os.path.exists(csv_path):
            with open(csv_path, "r", encoding="utf-8") as f:
                reader = csv.reader(f)
                next(reader, None)  # header
                for row in reader:
                    if len(row) < 3:
                        continue
                    abuse_type = row[0]
                    if abuse_type not in selected_types:
                        existing_rows.append(row)

        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["항목", "클러스터번호", "질문"])

            for row in existing_rows:
                writer.writerow(row)

            for row in self.cluster_details:
                writer.writerow([
                    row["abuse_type"],
                    row["cluster_id"],
                    row["question"]
                ])