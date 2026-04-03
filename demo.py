import fiftyone as fo
from pathlib import Path

dataset = fo.Dataset("crime-surveillance", overwrite=True)

video_dir = Path("data/test")  # adjust to your folder

for video_path in video_dir.glob("*.mp4"):
    # Extract label from filename: "Normal_Videos_006_x264.mp4" → "Normal"
    label = video_path.stem.split("_")[0]  # grabs first word before underscore
    
    sample = fo.Sample(filepath=str(video_path.resolve()))
    sample["ground_truth"] = fo.Classification(label=label)
    dataset.add_sample(sample)

dataset.compute_metadata()
dataset.persistent = True

print(dataset)
print(dataset.count_values("ground_truth.label"))

session = fo.launch_app(dataset)
session.wait()
