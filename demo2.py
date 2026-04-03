"""
FiftyOne Anomaly Detection Demo
=================================
Loads videos + pre-computed embeddings, runs anomaly detection and KNN
classification, displays results in FiftyOne App.

Usage:
    python fo_demo.py --video_dir data/segments \
                      --emb_dir results/pipeline \
                      --baseline_dir results/baseline \
                      --knn_path results/pipeline/knn_classifier.joblib
"""

import re
import json
import argparse
import numpy as np
import joblib
import fiftyone as fo
import fiftyone.brain as fob
from pathlib import Path
from anomaly_pipeline import cosine_similarity, cosine_distance_to_centroid, compute_z_scores


def extract_label(filename):
    match = re.match(r'^([A-Za-z]+)', Path(filename).stem)
    return match.group(1) if match else "Unknown"


def main():
    parser = argparse.ArgumentParser(description="FiftyOne Anomaly Detection Demo")
    parser.add_argument("--video_dir", required=True, help="Directory with video files")
    parser.add_argument("--emb_dir", required=True, help="Directory with per-video embedding JSONs")
    parser.add_argument("--baseline_dir", required=True, help="Directory with baseline_centroid.npy + baseline_stats.json")
    parser.add_argument("--knn_path", required=True, help="Path to knn_classifier.joblib")
    parser.add_argument("--dataset_name", default="crime-surveillance", help="FiftyOne dataset name")
    args = parser.parse_args()

    # ── Load pre-computed assets ───────────────────────
    print("Loading baseline...")
    centroid = np.load(Path(args.baseline_dir) / "baseline_centroid.npy").astype(np.float32)
    with open(Path(args.baseline_dir) / "baseline_stats.json") as f:
        stats = json.load(f)

    dist_mean = stats["normal_distance_mean"]
    dist_std = stats["normal_distance_std"]
    z_threshold = stats["z_threshold"]

    print("Loading KNN classifier...")
    knn = joblib.load(args.knn_path)
    print(f"  Classes: {list(knn.classes_)}")

    # ── Create dataset ─────────────────────────────────
    print("\nBuilding FiftyOne dataset...")
    dataset = fo.Dataset(args.dataset_name, overwrite=True)

    video_dir = Path(args.video_dir)
    emb_dir = Path(args.emb_dir)

    # ── Pass 1: Add all samples and compute metadata ───
    print("  Pass 1: Loading videos...")
    for video_path in sorted(video_dir.glob("*.mp4")):
        label = extract_label(video_path.stem)
        sample = fo.Sample(filepath=str(video_path.resolve()))
        sample["ground_truth"] = fo.Classification(label=label)
        dataset.add_sample(sample)

    dataset.compute_metadata()
    print(f"  Added {len(dataset)} videos")

    # ── Pass 2: Attach embeddings, anomaly scores, KNN ─
    print("  Pass 2: Computing anomaly scores + KNN...")
    videos_with_embeddings = 0

    for sample in dataset.iter_samples(autosave=True, progress=True):
        stem = Path(sample.filepath).stem
        emb_path = emb_dir / f"{stem}.json"

        if not emb_path.exists():
            continue

        with open(emb_path) as f:
            emb_data = json.load(f)

        # Get FPS from metadata (now available after compute_metadata)
        fps = 30.0
        if sample.metadata and sample.metadata.frame_rate:
            fps = sample.metadata.frame_rate

        detections = []
        z_scores = []
        len(emb_data["segments"])
        for seg in emb_data["segments"]:
            vec = np.array(seg["vector"], dtype=np.float32)

            # ── Anomaly score ──
            vec_norm = vec / (np.linalg.norm(vec) + 1e-9)
            cent_norm = centroid / (np.linalg.norm(centroid) + 1e-9)
            cos_dist = 1.0 - float(vec_norm @ cent_norm)
            z_score = (cos_dist - dist_mean) / max(dist_std, 1e-9)
            z_scores.append(z_score)

            # ── KNN classification ──
            vec_knn = vec.reshape(1, -1)
            vec_knn = vec_knn / (np.linalg.norm(vec_knn) + 1e-9)
            pred_label = knn.predict(vec_knn)[0]
            pred_conf = float(knn.predict_proba(vec_knn)[0].max())

            # ── Temporal detection (frame numbers) ──
            is_anomaly = z_score > z_threshold
            start_frame = max(1, int(seg["start_sec"] * fps) + 1)
            end_frame = max(start_frame, int(seg["end_sec"] * fps))

            det = fo.TemporalDetection(
                label=pred_label if is_anomaly else "Normal",
                support=[start_frame, end_frame],
                confidence=round(pred_conf, 4),
            )
            det["z_score"] = round(z_score, 4)
            det["cosine_distance"] = round(cos_dist, 6)
            det["is_anomaly"] = is_anomaly
            detections.append(det)

        # Write to sample
        sample["detections"] = fo.TemporalDetections(detections=detections)
        sample["max_z_score"] = round(max(z_scores), 4) if z_scores else 0.0
        sample["mean_z_score"] = round(float(np.mean(z_scores)), 4) if z_scores else 0.0
        sample["num_anomalies"] = sum(1 for d in detections if d["is_anomaly"])
        sample["num_segments"] = len(detections)

        # Video-level predicted label (from highest z-score segment)
        if detections:
            worst = max(detections, key=lambda d: d["z_score"])
            sample["predicted_crime"] = fo.Classification(
                label=worst.label,
                confidence=worst.confidence,
            )

        # Mean embedding for the brain visualizer
        all_vecs = np.array([s["vector"] for s in emb_data["segments"]], dtype=np.float32)
        sample["embedding"] = all_vecs.mean(axis=0).tolist()

        videos_with_embeddings += 1

    dataset.persistent = True

    print(f"\nLoaded {len(dataset)} videos ({videos_with_embeddings} with embeddings)")
    print(dataset)
    print("\nGround truth distribution:")
    print(dataset.count_values("ground_truth.label"))
    print("\nPredicted crime distribution:")
    try:
        print(dataset.count_values("predicted_crime.label"))
    except Exception:
        print("  (no predictions yet)")
    print(f"\nAnomaly summary:")
    print(f"  Videos with anomalies: {len(dataset.match(fo.ViewField('num_anomalies') > 0))}")

    # ── Compute brain visualization ────────────────────
    print("\nComputing embedding visualization...")
    try:
        fob.compute_visualization(
            dataset,
            embeddings="embedding",
            brain_key="marengo_viz",
            method="umap",
        )
        print("  Embedding visualization ready — click the brain icon in the App")
    except Exception as e:
        print(f"  Visualization failed (not critical): {e}")

    # ── Launch App ─────────────────────────────────────
    print("\nLaunching FiftyOne App...")
    print("  Tip: Sort by 'max_z_score' to see most anomalous videos first")
    print("  Tip: Filter 'num_anomalies > 0' to only see flagged videos")
    print("  Tip: Click brain icon → select 'marengo_viz' for embedding scatter plot")

    session = fo.launch_app(dataset)
    session.wait()


if __name__ == "__main__":
    main()