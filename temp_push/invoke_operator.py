import os
import fiftyone as fo
from twelvelabs import TwelveLabs
from voxel_ai_toolkit.backend import analyze_with_twelvelabs

key = os.environ.get("TWELVELABS_API_KEY")
if not key:
    print("NO_KEY")
    raise SystemExit(2)

client = TwelveLabs(api_key=key)

ds_name = "demo_voxel_test"
if ds_name not in fo.list_datasets():
    print("NO_DATASET")
    raise SystemExit(3)

print("Loading dataset:", ds_name)
ds = fo.load_dataset(ds_name)

processed = 0
for sample in ds:
    print("PROCESSING:", sample.filepath)
    try:
        res = analyze_with_twelvelabs(client, sample.filepath, action="summarize")
        print("RESULT:", res)
        if isinstance(res, dict) and res.get("summary"):
            sample["twelvelabs_summary"] = res["summary"]
            sample.save()
            print("SAVED summary for:", sample.filepath)
            processed += 1
    except Exception as e:
        print("ERROR processing", sample.filepath, str(e))

print(f"DONE: processed {processed} samples")
