import os
import json
import csv
from typing import Dict, List, Tuple
import numpy as np

def merge_centroid(
    data_dir,
    merge_threshold
):
    centroid_dir = os.path.join(data_dir, "embeddings", "centroids")
    labeled_dir = str(data_dir)
    details_dir = os.path.join(data_dir, "cluster_details")

    if not os.path.exists(centroid_dir):
        raise FileNotFoundError(f"Centroid directory not found: {centroid_dir}")

    merge_map, groups_map = build_full_merge_map(centroid_dir, merge_threshold)

    for abuse_type in sorted(groups_map.keys()):
        groups = groups_map[abuse_type]
        merged_groups = [g for g in groups if len(g) > 1]
        print(f"\n[{abuse_type}] merged clusters:")
        if merged_groups:
            for group in merged_groups:
                print(f"  - {group} -> {min(group)}")
        else:
            print("  - (no merges)")
        final_cluster_count = len(set(merge_map.get(abuse_type, {}).values()))
        print(f"[{abuse_type}] final cluster count: {final_cluster_count}")

    update_json_files(merge_map, labeled_dir)
    update_csv_files(merge_map, details_dir)
    print("Auto cluster merging completed successfully.")

    return merge_map


def build_full_merge_map(
    centroid_dir: str,
    similarity_threshold: float,
) -> Tuple[Dict[str, Dict[int, int]], Dict[str, List[List[int]]]]:
    full_map: Dict[str, Dict[int, int]] = {}
    groups_map: Dict[str, List[List[int]]] = {}
    for filename in os.listdir(centroid_dir):
        if not filename.endswith("_centroids.npy"):
            continue
        abuse_type = filename.replace("_centroids.npy", "")
        centroids = _load_centroids(os.path.join(centroid_dir, filename))
        merge_map, groups = _build_merge_map_for_type(
            centroids, similarity_threshold
        )
        full_map[abuse_type] = merge_map
        groups_map[abuse_type] = groups
    return full_map, groups_map


def update_json_files(
    merge_map: Dict[str, Dict[int, int]],
    labeled_dir: str,
):
    total_updated = 0
    for split in ["train", "test", "val"]:
        split_dir = os.path.join(labeled_dir, split)
        if not os.path.exists(split_dir):
            continue

        print(f"Updating JSON files in {split}...")
        for filename in os.listdir(split_dir):
            if not filename.endswith(".json"):
                continue

            filepath = os.path.join(split_dir, filename)
            with open(filepath, "r", encoding="utf-8") as f:
                try:
                    data = json.load(f)
                except Exception as e:
                    print(f"Error reading {filepath}: {e}")
                    continue

            changed = False
            for item in data.get("list", []):
                abuse_type = item.get("항목")
                if abuse_type in merge_map:
                    type_map = merge_map[abuse_type]
                    for audio in item.get("audio", []):
                        if "cluster" in audio:
                            cid = audio["cluster"]
                            if cid in type_map:
                                new_cid = type_map[cid]
                                if cid != new_cid:
                                    audio["cluster"] = int(new_cid)
                                    changed = True

            if changed:
                with open(filepath, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
                total_updated += 1

    print(f"Total JSON files updated: {total_updated}")


def update_csv_files(
    merge_map: Dict[str, Dict[int, int]],
    details_dir: str,
):
    for csv_file in [
        "unique_cluster_details.csv",
        "test_cluster_details.csv",
        "val_cluster_details.csv",
    ]:
        csv_path = os.path.join(details_dir, csv_file)
        if not os.path.exists(csv_path):
            continue

        print(f"Updating CSV file: {csv_file}")
        rows = []
        updated_count = 0
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            if header:
                rows.append(header)
                for row in reader:
                    if len(row) < 2:
                        rows.append(row)
                        continue

                    abuse_type = row[0]
                    try:
                        cluster_id = int(row[1])
                    except ValueError:
                        rows.append(row)
                        continue

                    if abuse_type in merge_map:
                        type_map = merge_map[abuse_type]
                        if cluster_id in type_map:
                            new_cid = type_map[cluster_id]
                            if cluster_id != new_cid:
                                row[1] = str(new_cid)
                                updated_count += 1
                    rows.append(row)

        if updated_count > 0:
            with open(csv_path, "w", encoding="utf-8", newline="") as f:
                writer = csv.writer(f)
                writer.writerows(rows)
            print(f"  Updated {updated_count} rows in {csv_file}")
        else:
            print(f"  No changes needed in {csv_file}")


def _load_centroids(centroid_path: str) -> Dict[int, np.ndarray]:
    centroids_obj = np.load(centroid_path, allow_pickle=True)
    if isinstance(centroids_obj, np.ndarray) and centroids_obj.shape == ():
        centroids = centroids_obj.item()
    else:
        centroids = centroids_obj
    return {int(k): np.asarray(v) for k, v in centroids.items()}


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def _build_union_find(items: List[int]):
    parent = {i: i for i in items}
    rank = {i: 0 for i in items}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra == rb:
            return
        if rank[ra] < rank[rb]:
            parent[ra] = rb
        elif rank[ra] > rank[rb]:
            parent[rb] = ra
        else:
            parent[rb] = ra
            rank[ra] += 1

    return parent, find, union


def _build_merge_map_for_type(
    centroids: Dict[int, np.ndarray],
    similarity_threshold: float,
) -> Tuple[Dict[int, int], List[List[int]]]:
    cluster_ids = sorted(centroids.keys())
    if len(cluster_ids) <= 1:
        return {cid: cid for cid in cluster_ids}, [cluster_ids] if cluster_ids else []

    centroid_matrix = np.vstack([centroids[cid] for cid in cluster_ids])
    centroid_matrix = _normalize_rows(centroid_matrix)
    sim_matrix = centroid_matrix @ centroid_matrix.T

    parent, find, union = _build_union_find(cluster_ids)

    for i in range(len(cluster_ids)):
        for j in range(i + 1, len(cluster_ids)):
            if sim_matrix[i, j] >= similarity_threshold:
                union(cluster_ids[i], cluster_ids[j])

    components: Dict[int, List[int]] = {}
    for cid in cluster_ids:
        root = find(cid)
        components.setdefault(root, []).append(cid)

    merge_map: Dict[int, int] = {}
    for group in components.values():
        target = min(group)
        for cid in group:
            merge_map[cid] = target

    groups = [sorted(group) for group in components.values()]
    groups.sort(key=lambda g: (len(g) == 1, g))
    return merge_map, groups
