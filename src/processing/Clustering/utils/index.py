
import os
import json
import numpy as np
import csv
from sklearn.metrics.pairwise import cosine_similarity

def embed_and_assign_to_clusters(
    data_dir,
    centroids_data,
    model,
    data_type,
    similarity_threshold
):
    os.makedirs(os.path.join(data_dir, data_type), exist_ok=True)
    data_dir = os.path.join(data_dir, data_type)
    # 유사도 통계
    all_max_similarities = []

    # test cluster 상세 저장용
    test_cluster_details = []

    for filename in os.listdir(data_dir):
        if not filename.endswith('.json'):
            continue

        output_path = os.path.join(data_dir, filename)
        with open(output_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        for item_idx, item in enumerate(data.get('list', [])):
            abuse_type = item.get('항목')
            if abuse_type not in centroids_data:
                continue

            questions = []
            audio_indexes = []

            for audio_idx, audio in enumerate(item.get('audio', [])):
                if audio.get('type') == 'Q':
                    if audio.get('text') == '|end|':
                        audio['cluster'] = -2
                        continue
                    questions.append(audio['text'])
                    audio_indexes.append(audio_idx)

            if not questions:
                continue

            embeddings = model.encode(
                questions,
                batch_size=32,
                show_progress_bar=False,
                normalize_embeddings=True
            )

            centroids_dict = centroids_data[abuse_type]

            clusters, max_sims = assign_clusters_with_stats(
                embeddings,
                centroids_dict,
                similarity_threshold
            )

            all_max_similarities.extend(max_sims.tolist())

            # JSON 반영 + CSV 기록
            for audio_idx, cluster_id, question in zip(audio_indexes, clusters, questions):
                item['audio'][audio_idx]['cluster'] = int(cluster_id)

                test_cluster_details.append({
                    "abuse_type": abuse_type,
                    "cluster_id": int(cluster_id),
                    "question": question
                })

        # 파일 저장
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    # 📈 유사도 통계 출력
    if all_max_similarities:
        all_max_similarities = np.array(all_max_similarities)
        print("\n" + "=" * 50)
        print("FINAL SIMILARITY ANALYSIS REPORT")
        print("=" * 50)
        print(f"Total Questions Processed : {len(all_max_similarities)}")
        print(f"Mean Similarity           : {np.mean(all_max_similarities):.4f}")
        print(f"Median Similarity         : {np.median(all_max_similarities):.4f}")
        print(f"Min Similarity            : {np.min(all_max_similarities):.4f}")
        print(f"Max Similarity            : {np.max(all_max_similarities):.4f}")
        print(f"Threshold (Current)       : {similarity_threshold}")

        assigned_count = np.sum(all_max_similarities >= similarity_threshold)
        fail_count = len(all_max_similarities) - assigned_count
        print(f"Assigned Clusters         : {assigned_count} ({assigned_count/len(all_max_similarities)*100:.1f}%)")
        print(f"Unassigned (-1)           : {fail_count} ({fail_count/len(all_max_similarities)*100:.1f}%)")
        print("=" * 50 + "\n")

    # save_test_cluster_details_csv(
    #     test_cluster_details,
    #     data_dir,
    #     data_type
    # )


def assign_clusters_with_stats(embeddings, centroids_dict, threshold):
    cluster_ids = np.array(list(centroids_dict.keys()))
    centroid_matrix = np.vstack([centroids_dict[cid] for cid in cluster_ids])
    similarities = cosine_similarity(embeddings, centroid_matrix)
    best_indices = np.argmax(similarities, axis=1)
    max_sims = np.max(similarities, axis=1)
    results = np.full(len(embeddings), -1, dtype=int)
    mask = max_sims >= threshold
    results[mask] = cluster_ids[best_indices[mask]]

    return results, max_sims


def save_test_cluster_details_csv(cluster_details, output_directory, test_or_val):
    cluster_details.sort(
        key=lambda x: (x["abuse_type"], x["cluster_id"])
    )

    out_dir = os.path.join(output_directory, "cluster_details")
    os.makedirs(out_dir, exist_ok=True)

    csv_path = os.path.join(out_dir, f"{test_or_val}_cluster_details.csv")

    existing_rows = []
    if os.path.exists(csv_path):
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader, None)  # header
            for row in reader:
                if len(row) < 3:
                    continue
                existing_rows.append(row)

    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["항목", "클러스터번호", "질문"])

        for row in existing_rows:
            writer.writerow(row)

        for row in cluster_details:
            writer.writerow([
                row["abuse_type"],
                row["cluster_id"],
                row["question"]
            ])