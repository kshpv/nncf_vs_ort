from pathlib import Path

from download_dataset import download_file
from PIL import Image
from ultralytics import YOLO

# Download a test sample
IMAGE_PATH = Path('./data/coco_bike.jpg')
download_file(
    url='https://storage.openvinotoolkit.org/repositories/openvino_notebooks/data/data/image/coco_bike.jpg',
    filename=IMAGE_PATH.name,
    directory=IMAGE_PATH.parent
) 


models_dir = Path('./models')
models_dir.mkdir(exist_ok=True)

DET_MODEL_NAME = "yolov8n"

det_model = YOLO(models_dir / f'{DET_MODEL_NAME}.pt')
label_map = det_model.model.names

res = det_model(IMAGE_PATH)
Image.fromarray(res[0].plot()[:, :, ::-1])

# object detection model
det_model.export(format="openvino", dynamic=True, half=False)