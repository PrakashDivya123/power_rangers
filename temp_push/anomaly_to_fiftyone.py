import json
import csv
import fiftyone as fo
import os

DATA_DIR = r"D:\VoxelHackathon\anomaly_baseline-20260403T193608Z-3-001\anomaly_baseline"
DATASET_NAME = "anomaly_baseline_demo"

# Placeholder image to attach so the App can show a sample
PLACEHOLDER_IMAGE = "https://upload.wikimedia.org/wikipedia/commons/thumb/4/47/PNG_transparency_demonstration_1.png/320px-PNG_transparency_demonstration_1.png"

# Read baseline_stats.json
stats_path = os.path.join(DATA_DIR, "baseline_stats.json")
scores_path = os.path.join(DATA_DIR, "training_segment_scores.csv")

with open(stats_path, "r", encoding="utf8") as f:
    baseline_stats = json.load(f)

# Read scores CSV into a list
scores = []
with open(scores_path, newline='', encoding='utf8') as csvfile:
    reader = csv.reader(csvfile)
    # assume first row is header; collect numeric values from first data column
    rows = list(reader)
    if len(rows) > 1:
        for r in rows[1:]:
            try:
                scores.append(float(r[0]))
            except Exception:
                # try to parse all numeric columns
                for v in r:
                    try:
                        scores.append(float(v))
                    except Exception:
                        pass

# Create or replace dataset
if fo.dataset_exists(DATASET_NAME):
    fo.delete_dataset(DATASET_NAME)

dataset = fo.Dataset(DATASET_NAME)

sample = fo.Sample(filepath=PLACEHOLDER_IMAGE)
# Attach baseline stats and scores as fields
sample['baseline_stats'] = baseline_stats
sample['training_scores'] = scores

dataset.add_sample(sample)

print(f"Created dataset: {DATASET_NAME}")
print(f"Samples: {dataset.count()}")
print(f"Baseline stats keys: {list(baseline_stats.keys())}")
print(f"First 10 training scores: {scores[:10]}")
