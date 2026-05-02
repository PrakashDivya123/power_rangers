# Remove top-level baseline load; this caused import-time errors when
# `np` and `BASELINE_DIR` were not yet defined. Baseline is loaded
# on-demand inside functions that require it.
"""Command-line utility for preprocessing a video, uploading to TwelveLabs,
creating clip embeddings (Marengo), and optionally scoring against a baseline.

This file is a refactor of the original Colab notebook into a reusable script
for local/Windows use. API key must be provided via the TWELVELABS_API_KEY
environment variable.
"""

from pathlib import Path
import os
import sys
import json
import time
import shutil
import subprocess
import argparse
from typing import List, Optional

import numpy as np
import pandas as pd

try:
    from twelvelabs import TwelveLabs, VideoInputRequest, MediaSource
except Exception:
    TwelveLabs = None

VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}
ASSET_POLL_SECONDS = 3
MAX_ASSET_WAIT_SECONDS = 600
EMBEDDING_OPTIONS = ["visual"]
USE_FUSED = False


def cosine_similarity(a, b):
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)

    if a.ndim == 1:
        a = a.reshape(1, -1)
    if b.ndim == 1:
        b = b.reshape(1, -1)

    a_norm = np.linalg.norm(a, axis=1, keepdims=True)
    b_norm = np.linalg.norm(b, axis=1, keepdims=True)

    a_norm = np.clip(a_norm, 1e-12, None)
    b_norm = np.clip(b_norm, 1e-12, None)

    a_unit = a / a_norm
    b_unit = b / b_norm

    return a_unit @ b_unit.T


def l2_normalize_rows(x, eps=1e-12):
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.clip(norms, eps, None)


def l2_normalize_vec(x, eps=1e-12):
    norm = np.linalg.norm(x)
    return x / max(norm, eps)


def cosine_distance_to_centroid(embeddings, centroid):
    embeddings = l2_normalize_rows(np.asarray(embeddings, dtype=np.float32))
    centroid = l2_normalize_vec(np.asarray(centroid, dtype=np.float32)).reshape(1, -1)
    sims = cosine_similarity(embeddings, centroid).reshape(-1)
    return 1.0 - sims


def compute_z_scores(values, ref_mean, ref_std, eps=1e-12):
    return (values - ref_mean) / max(ref_std, eps)


def ffprobe_resolution(video_path: str):
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height",
        "-of", "json",
        video_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    data = json.loads(result.stdout)
    stream = data["streams"][0]
    return int(stream["width"]), int(stream["height"])


def preprocess_video_for_twelvelabs(src_path: str, dst_path: str, min_size: int = 360):
    w, h = ffprobe_resolution(src_path)

    os.makedirs(os.path.dirname(dst_path), exist_ok=True)

    if w >= min_size and h >= min_size:
        shutil.copy2(src_path, dst_path)
        return {"action": "copied", "width": w, "height": h}

    vf = (
        f"scale=w='max(iw,{min_size})':h='max(ih,{min_size})':force_original_aspect_ratio=decrease,"
        f"pad={min_size}:{min_size}:(ow-iw)/2:(oh-ih)/2"
    )

    cmd = [
        "ffmpeg", "-y",
        "-i", src_path,
        "-vf", vf,
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "23",
        "-c:a", "aac",
        "-movflags", "+faststart",
        dst_path
    ]
    subprocess.run(cmd, check=True)
    return {"action": "resized_or_padded", "width": w, "height": h}


def upload_asset(client, local_video_path: str):
    with open(local_video_path, "rb") as f:
        asset = client.assets.create(method="direct", file=f, filename=os.path.basename(local_video_path))

    asset_id = asset.id
    start_t = time.time()

    while True:
        asset_status = client.assets.retrieve(asset_id=asset_id)
        status = getattr(asset_status, "status", None)

        if status == "ready":
            return asset_status
        if status == "failed":
            raise RuntimeError(f"Asset processing failed for {local_video_path}")
        if time.time() - start_t > MAX_ASSET_WAIT_SECONDS:
            raise TimeoutError(f"Timed out waiting for asset: {local_video_path}")

        time.sleep(ASSET_POLL_SECONDS)


def create_clip_embeddings_from_asset(client, asset_id: str, segment_seconds: int = 10):
    video_request = VideoInputRequest(
        media_source=MediaSource(asset_id=asset_id),
        segmentation={"strategy": "fixed", "fixed": {"duration_sec": segment_seconds}},
        embedding_option=EMBEDDING_OPTIONS,
        embedding_scope=["clip"],
        **({"embedding_type": ["fused_embedding"]} if USE_FUSED else {}),
    )

    response = client.embed.v_2.create(input_type="video", model_name="marengo3.0", video=video_request)

    records = []
    for item in response.data:
        if getattr(item, "embedding_scope", None) != "clip":
            continue

        option = getattr(item, "embedding_option", None)
        if USE_FUSED and option != "fused":
            continue

        records.append({
            "start_sec": float(getattr(item, "start_sec", 0.0) or 0.0),
            "end_sec": float(getattr(item, "end_sec", 0.0) or 0.0),
            "embedding_option": option,
            "embedding": np.array(item.embedding, dtype=np.float32),
        })

    if not records:
        raise RuntimeError("No clip embeddings returned")
    return records


def analyze_with_twelvelabs(client, local_video_path: str, action: str = "summarize", prompt: Optional[str] = None, umbrella: Optional[str] = None, classes: Optional[List[str]] = None, occluded: bool = False):
    """Upload a video via the provided `client` and run a simple Twelve Labs analysis.

    Returns a dict: for `summarize` -> {"summary": text}, for `zero_shot_classify` -> {"classification": label}.
    """
    if TwelveLabs is None:
        raise RuntimeError("twelvelabs package is not installed")

    # Upload asset and wait until processed
    asset_status = upload_asset(client, local_video_path)
    asset_id = getattr(asset_status, "id", None)
    if not asset_id:
        raise RuntimeError("Failed to upload asset or retrieve asset id")

    if action == "summarize":
        the_prompt = prompt or "Summarize this video clip."
        res = client.generate.text(video_id=asset_id, prompt=the_prompt)
        return {"summary": getattr(res, "data", None)}

    if action == "zero_shot_classify":
        umbrella = umbrella or "Anomaly"
        classes = classes or []
        class_list_str = ", ".join(classes)
        the_prompt = f"Given the umbrella category '{umbrella}', which of these classes best categorizes this video: [{class_list_str}]? Respond only with the class name."
        if occluded:
            the_prompt += " Focus on unnatural movements, lower-body motion, or individuals concealing themselves."

        res = client.generate.text(video_id=asset_id, prompt=the_prompt)
        return {"classification": getattr(res, "data", "").strip()}

    return {"status": "unsupported_action"}


def list_video_files(folder: str) -> List[str]:
    p = Path(folder)
    if not p.exists():
        return []
    return sorted([str(x) for x in p.rglob("*") if x.suffix.lower() in VIDEO_EXTS])


def export_top_clips(original_video: str, rows: pd.DataFrame, out_dir: str, top_k: int = 5):
    os.makedirs(out_dir, exist_ok=True)
    exported = []
    for _, row in rows.head(top_k).iterrows():
        start_sec = float(row["start_sec"])
        end_sec = float(row["end_sec"])
        duration = max(0.1, end_sec - start_sec)
        video_name = Path(original_video).stem
        clip_path = os.path.join(out_dir, f"{video_name}_rank{int(row['rank']):02d}_seg{int(row['segment_index']):03d}_z{row['z_score']:.2f}.mp4")
        cmd = [
            "ffmpeg", "-y",
            "-ss", str(start_sec),
            "-i", original_video,
            "-t", str(duration),
            "-c:v", "libx264",
            "-c:a", "aac",
            "-movflags", "+faststart",
            clip_path,
        ]
        subprocess.run(cmd, check=True)
        exported.append(clip_path)
    return exported


def process_video(
    video_path: str,
    output_dir: str,
    api_key: str,
    segment_seconds: int = 10,
    baseline_dir: Optional[str] = None,
    top_k: int = 5,
):
    if TwelveLabs is None:
        raise RuntimeError("twelvelabs package is not installed")

    client = TwelveLabs(api_key=api_key)

    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    prep_dir = os.path.join(output_dir, "preprocessed")
    clips_dir = os.path.join(output_dir, "anomaly_clips")
    merge_dir = os.path.join(output_dir, "merged_videos")
    json_dir = os.path.join(output_dir, "results")
    for d in (prep_dir, clips_dir, merge_dir, json_dir):
        os.makedirs(d, exist_ok=True)

    video_name = Path(video_path).stem
    prep_video = os.path.join(prep_dir, os.path.basename(video_path))

    meta = preprocess_video_for_twelvelabs(video_path, prep_video, min_size=360)
    print("Preprocess:", meta)

    asset = upload_asset(client, prep_video)
    clip_records = create_clip_embeddings_from_asset(client, asset.id, segment_seconds=segment_seconds)

    if len(clip_records) == 0:
        raise RuntimeError("No clip embeddings returned for video")

    embeddings = np.vstack([r["embedding"] for r in clip_records]).astype(np.float32)

    result_df = pd.DataFrame({
        "video_path": [video_path] * len(clip_records),
        "video_name": [os.path.basename(video_path)] * len(clip_records),
        "asset_id": [asset.id] * len(clip_records),
        "segment_index": list(range(len(clip_records))),
        "start_sec": [r["start_sec"] for r in clip_records],
        "end_sec": [r["end_sec"] for r in clip_records],
        "duration_sec": [r["end_sec"] - r["start_sec"] for r in clip_records],
        "embedding_option": [r["embedding_option"] for r in clip_records],
    })

    if baseline_dir:
        baseline_centroid = np.load(os.path.join(baseline_dir, "baseline_centroid.npy")).astype(np.float32)
        with open(os.path.join(baseline_dir, "baseline_stats.json"), "r") as f:
            baseline_stats = json.load(f)
        SEGMENT_SECONDS = int(baseline_stats.get("segment_seconds", segment_seconds))
        Z_THRESHOLD = float(baseline_stats.get("z_threshold", 3.0))
        dist_mean = float(baseline_stats.get("normal_distance_mean", 0.0))
        dist_std = float(baseline_stats.get("normal_distance_std", 1.0))

        cos_dist = cosine_distance_to_centroid(embeddings, baseline_centroid)
        z = compute_z_scores(cos_dist, dist_mean, dist_std)

        result_df["cosine_distance"] = cos_dist
        result_df["z_score"] = z
        result_df["is_anomaly"] = z >= Z_THRESHOLD
    else:
        result_df["cosine_distance"] = np.nan
        result_df["z_score"] = np.nan
        result_df["is_anomaly"] = False

    result_df = result_df.sort_values("z_score", ascending=False, na_position="last").reset_index(drop=True)
    result_df["rank"] = np.arange(1, len(result_df) + 1)

    csv_out = os.path.join(json_dir, f"{video_name}_segment_scores.csv")
    json_out = os.path.join(json_dir, f"{video_name}_segment_scores.json")
    npy_out = os.path.join(json_dir, f"{video_name}_segment_embeddings.npy")

    result_df.to_csv(csv_out, index=False)
    result_df.to_json(json_out, orient="records", indent=2)
    np.save(npy_out, embeddings)

    print("\nTop scored segments:")
    print(result_df.head(top_k).to_string(index=False))

    exported = []
    if baseline_dir:
        exported = export_top_clips(video_path, result_df, clips_dir, top_k=top_k)
        print(f"\nSaved {len(exported)} clips to: {clips_dir}")

    return {
        "csv": csv_out,
        "json": json_out,
        "npy": npy_out,
        "clips": exported,
    }


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("video", help="Path to input video file")
    p.add_argument("--output", default="./output_anomaly", help="Output directory")
    p.add_argument("--segment-seconds", type=int, default=10)
    p.add_argument("--baseline-dir", default=None, help="Path to baseline directory (optional)")
    p.add_argument("--top-k", type=int, default=5)
    return p.parse_args()


def main():
    args = parse_args()
    api_key = os.environ.get("TWELVELABS_API_KEY")
    if not api_key:
        print("Error: set TWELVELABS_API_KEY in environment", file=sys.stderr)
        sys.exit(2)

    res = process_video(
        video_path=args.video,
        output_dir=args.output,
        api_key=api_key,
        segment_seconds=args.segment_seconds,
        baseline_dir=args.baseline_dir,
        top_k=args.top_k,
    )

    print("Results:")
    print(res)


if __name__ == "__main__":
    main()

