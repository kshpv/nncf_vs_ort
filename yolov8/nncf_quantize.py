from pathlib import Path
from typing import Dict

import onnx
from ultralytics import YOLO
from ultralytics.yolo.cfg import get_cfg
from ultralytics.yolo.data.utils import check_det_dataset
from ultralytics.yolo.utils import DEFAULT_CFG
from ultralytics.yolo.utils import ops

import nncf  # noqa: F811

DET_MODEL_NAME = "yolov8n"
models_dir = Path('./models')
det_model_path = models_dir / f"{DET_MODEL_NAME}.onnx"
det_model = YOLO(models_dir / f'{DET_MODEL_NAME}.pt')

OUT_DIR = Path('./datasets')

DATA_PATH = OUT_DIR / "val2017.zip"
LABELS_PATH = OUT_DIR / "coco2017labels-segments.zip"
CFG_PATH = OUT_DIR / "coco.yaml"

args = get_cfg(cfg=DEFAULT_CFG)
args.data = str(CFG_PATH)

det_validator = det_model.ValidatorClass(args=args)

det_validator.data = check_det_dataset(args.data)
det_data_loader = det_validator.get_dataloader("datasets/coco", 1)

det_validator.is_coco = True
det_validator.class_map = ops.coco80_to_coco91_class()
det_validator.names = det_model.model.names
det_validator.metrics.names = det_validator.names
det_validator.nc = det_model.model.model[-1].nc


def transform_fn(data_item:Dict):
    """
    Quantization transform function. Extracts and preprocess input data from dataloader item for quantization.
    Parameters:
       data_item: Dict with data item produced by DataLoader during iteration
    Returns:
        input_tensor: Input data for quantization
    """
    input_tensor = det_validator.preprocess(data_item)['img'].numpy()
    return {'images': input_tensor}

det_data_loader = det_validator.get_dataloader("datasets/coco", 1)

quantization_dataset = nncf.Dataset(det_data_loader, transform_fn)

ignored_scope = nncf.IgnoredScope(
    types=["Mul", "Sub", "Sigmoid"],  # ignore operations
    names=[
        "/model.22/dfl/conv/Conv",           # in the post-processing subgraph
        "/model.22/Add",
        "/model.22/Add_1",
        "/model.22/Add_2",
        "/model.22/Add_3",
        "/model.22/Add_4",   
        "/model.22/Add_5",
        "/model.22/Add_6",
        "/model.22/Add_7",
        "/model.22/Add_8",
        "/model.22/Add_9",
        "/model.22/Add_10"
    ]
)

DET_MODEL_NAME = "yolov8n"
models_dir = Path('./models')
det_model_path = models_dir / f"{DET_MODEL_NAME}.onnx"

det_onnx_model = onnx.load(det_model_path)

# Detection model
quantized_det_model = nncf.quantize(
    det_onnx_model,
    quantization_dataset,
    preset=nncf.QuantizationPreset.MIXED,
    ignored_scope=ignored_scope
)

int8_model_path = models_dir / f"{DET_MODEL_NAME}_int8_NNCF.onnx"
onnx.save_model(quantized_det_model, int8_model_path)