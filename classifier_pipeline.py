"""
Pipeline: Extract labels → Compute embeddings → Train KNN
==========================================================
Usage:
    python pipeline.py --video_dir data/segments --output_dir results/pipeline
"""

import os
import re
import json
import numpy as np
import argparse
from pathlib import Path
from dotenv import load_dotenv
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

load_dotenv()

# ── Step 1: Extract labels from filenames ──────────────

def extract_label(filename):
    """
    Extract class label from filename.
    'Abuse001_x264_rank01_seg003_z84.19.mp4'    → 'Abuse'
    'Arrest006_x264_rank01_seg001_z3.07.mp4'    → 'Arrest'
    'Normal_Videos_014_x264_rank01_seg004.mp4'  → 'Normal'
    """
    name = Path(filename).stem
    # Match letters at the start before the first digit
    match = re.match(r'^([A-Za-z]+)', name)
    if match:
        label = match.group(1)
        # Handle "Normal_Videos" → "Normal" (already handled since we only grab letters before digits)
        return label
    return "Unknown"


def build_file_manifest(video_dir):
    """Scan directory and build list of (filepath, label) tuples."""
    video_dir = Path(video_dir)
    manifest = []
    for f in sorted(video_dir.glob("*.mp4")):
        label = extract_label(f.name)
        manifest.append({"filepath": str(f), "filename": f.name, "label": label})

    # Print summary
    from collections import Counter
    counts = Counter(m["label"] for m in manifest)
    print(f"Found {len(manifest)} videos:")
    for label, count in sorted(counts.items()):
        print(f"  {label}: {count}")

    return manifest


# ── Step 2: Compute embeddings ─────────────────────────

def compute_embeddings_batch(manifest, output_dir):
    """
    Compute Marengo embeddings for all videos.
    Saves per-video embedding JSON files + a combined numpy array.
    """
    import base64
    from twelvelabs import TwelveLabs, VideoInputRequest, MediaSource
    import time

    client = TwelveLabs(api_key=os.getenv("TWELVE_LABS_API_KEY"))
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_vectors = []
    all_labels = []
    failed = []

    for i, item in enumerate(manifest):
        filepath = item["filepath"]
        label = item["label"]
        stem = Path(filepath).stem
        emb_path = output_dir / f"{stem}.json"

        # Skip if already computed
        if emb_path.exists():
            with open(emb_path) as f:
                data = json.load(f)
            for seg in data["segments"]:
                all_vectors.append(seg["vector"])
                all_labels.append(label)
            print(f"  [{i+1}/{len(manifest)}] {stem} — cached ({len(data['segments'])} segments)")
            continue

        print(f"  [{i+1}/{len(manifest)}] {stem} — embedding...", end=" ")

        try:
            with open(filepath, "rb") as vf:
                video_b64 = base64.b64encode(vf.read()).decode("ascii")

            task = client.embed.v_2.tasks.create(
                input_type="video",
                model_name="marengo3.0",
                video=VideoInputRequest(
                    media_source=MediaSource(base_64_string=video_b64),
                ),
            )

            # Poll
            while True:
                task = client.embed.v_2.tasks.retrieve(task_id=task.id)
                if task.status == "ready":
                    break
                elif task.status in ("failed", "error"):
                    raise RuntimeError(f"Task failed: {task.status}")
                time.sleep(3)

            segments = []
            if task.data:
                for seg in task.data:
                    segments.append({
                        "start_sec": seg.start_sec,
                        "end_sec": seg.end_sec,
                        "scope": seg.embedding_scope,
                        "option": seg.embedding_option,
                        "vector": seg.embedding,
                    })
                    all_vectors.append(seg.embedding)
                    all_labels.append(label)

            # Save per-video
            with open(emb_path, "w") as f:
                json.dump({"filename": item["filename"], "label": label, "segments": segments}, f)

            print(f"✓ {len(segments)} segments")

        except Exception as e:
            print(f"✗ {e}")
            failed.append({"filename": item["filename"], "error": str(e)})

    # Save combined arrays
    X = np.array(all_vectors)
    y = np.array(all_labels)
    np.save(output_dir / "X_embeddings.npy", X)
    np.save(output_dir / "y_labels.npy", y)

    print(f"\nDone: {len(all_vectors)} total segments, {len(failed)} failed")
    if failed:
        print("Failed videos:")
        for f in failed:
            print(f"  {f['filename']}: {f['error']}")

    return X, y


# ── Step 3: Train KNN ─────────────────────────────────

def train_knn(X, y, output_dir, k=5):
    """Train KNN classifier and save it."""
    output_dir = Path(output_dir)

    # Normalize for cosine
    norms = np.linalg.norm(X, axis=1, keepdims=True) + 1e-9
    X_norm = X / norms

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_norm, y, test_size=0.2, random_state=42, stratify=y
    )

    # Train
    knn = KNeighborsClassifier(n_neighbors=k, metric="cosine", weights="distance")
    knn.fit(X_train, y_train)

    # Evaluate
    y_pred = knn.predict(X_test)
    print("\n" + "=" * 60)
    print("KNN Classification Report")
    print("=" * 60)
    print(classification_report(y_test, y_pred))

    accuracy = (y_pred == y_test).mean()
    print(f"Accuracy: {accuracy:.4f}")

    # Save
    model_path = output_dir / "knn_classifier.joblib"
    joblib.dump(knn, model_path)
    print(f"Model saved: {model_path}")

    # Also save class list
    classes = sorted(set(y))
    with open(output_dir / "classes.json", "w") as f:
        json.dump(classes, f)

    return knn


# ── Step 4: Quick inference helper ─────────────────────

def classify_video(knn, segment_embedding):
    """Classify a single segment embedding."""
    vec = np.array(segment_embedding).reshape(1, -1)
    vec = vec / (np.linalg.norm(vec) + 1e-9)
    label = knn.predict(vec)[0]
    proba = knn.predict_proba(vec)[0]
    confidence = float(proba.max())
    return label, round(confidence, 4)


# ── CLI ────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Extract labels, embed videos, train KNN")
    parser.add_argument("--video_dir", required=True, help="Directory with video segments")
    parser.add_argument("--output_dir", default="results/pipeline", help="Where to save embeddings + model")
    parser.add_argument("--k", type=int, default=5, help="K for KNN")
    parser.add_argument("--skip_embed", action="store_true", help="Skip embedding, load from disk")
    args = parser.parse_args()

    # Step 1: Extract labels
    print("Step 1: Scanning videos...")
    manifest = build_file_manifest(args.video_dir)

    # Step 2: Compute embeddings
    output_dir = Path(args.output_dir)
    if args.skip_embed:
        print("\nStep 2: Loading cached embeddings...")
        X = np.load(output_dir / "X_embeddings.npy")
        y = np.load(output_dir / "y_labels.npy")
        print(f"  Loaded {len(X)} segments")
    else:
        print("\nStep 2: Computing embeddings...")
        X, y = compute_embeddings_batch(manifest, output_dir)

    # Step 3: Train KNN
    print("\nStep 3: Training KNN...")
    
    knn = train_knn(X, y, output_dir, k=args.k)


if __name__ == "__main__":
    main()