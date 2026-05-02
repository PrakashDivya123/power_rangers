import fiftyone as fo
import fiftyone.utils.data as fud

# Create or replace a small demo dataset with one image
DATASET_NAME = "demo_dataset"

if fo.dataset_exists(DATASET_NAME):
    fo.delete_dataset(DATASET_NAME)

# Use a small public image URL
image_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/4/47/PNG_transparency_demonstration_1.png/320px-PNG_transparency_demonstration_1.png"
sample = fo.Sample(filepath=image_url)

dataset = fo.Dataset(DATASET_NAME)
dataset.add_sample(sample)

print(f"Created dataset: {DATASET_NAME}")
print(f"Samples in dataset: {dataset.count_samples()}")
for s in dataset:
    print(s.id, s.filepath)

# Optionally launch the app automatically (commented out by default)
# fo.launch_app(dataset=dataset)
