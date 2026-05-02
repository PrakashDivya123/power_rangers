import fiftyone as fo
import numpy as np, os, tempfile, shutil, sys, time

try:
    print("FO_VERSION", fo.__version__)
except Exception as e:
    print("FIFTYONE_IMPORT_ERROR", e)
    sys.exit(2)

workdir = tempfile.mkdtemp(prefix="voxelfo_")
print("WORKDIR", workdir)

try:
    from PIL import Image
except Exception:
    print("PIL_MISSING")
    sys.exit(3)

img1 = (np.zeros((64,64,3),dtype=np.uint8)+255)
img2 = (np.zeros((64,64,3),dtype=np.uint8)+128)

p1 = os.path.join(workdir, "a.png")
p2 = os.path.join(workdir, "b.png")
Image.fromarray(img1).save(p1)
Image.fromarray(img2).save(p2)

ds_name = "demo_voxel_test"
if ds_name in fo.list_datasets():
    fo.delete_dataset(ds_name)

ds = fo.Dataset(ds_name)
ds.add_samples([fo.Sample(filepath=p1), fo.Sample(filepath=p2)])
print("DATASET_CREATED", ds_name, len(ds))

app = fo.launch_app(dataset=ds, port=5151)
# app.url attribute exists in newer versions; otherwise default
url = getattr(app, "url", "http://localhost:5151")
print("APP_URL", url)
print("Launched FiftyOne app. Keeping process alive for 10 minutes for demo.")
try:
    time.sleep(600)
except KeyboardInterrupt:
    print("Interrupted by user; exiting.")

try:
    app.close()
except Exception:
    pass

try:
    fo.delete_dataset(ds_name)
except Exception:
    pass

shutil.rmtree(workdir, ignore_errors=True)
print("CLEANED_UP")
