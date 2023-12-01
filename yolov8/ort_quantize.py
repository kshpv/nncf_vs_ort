from pathlib import Path

import onnx
import onnxruntime as rt
from onnxruntime.quantization import CalibrationDataReader
from onnxruntime.quantization import QuantFormat
from onnxruntime.quantization import quantize_static
from ultralytics import YOLO
from ultralytics.yolo.cfg import get_cfg
from ultralytics.yolo.data.utils import check_det_dataset
from ultralytics.yolo.utils import DEFAULT_CFG
from ultralytics.yolo.utils import ops


class MyData(CalibrationDataReader):
    def __init__(self, data_validator, num_samples=300) -> None:
        super().__init__()
        self.data_validator = data_validator
        self.num_samples = num_samples
        self.iter_data_validator = iter(self.data_validator)
        self.i = 0
        
    
    def get_next(self) -> dict:
        print (self.i)
        if self.i == 300:
            return None
        input_tensor = det_validator.preprocess(next(self.iter_data_validator, None))['img'].numpy()
        self.i += 1
        return {'images': input_tensor}


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

det_data_loader = det_validator.get_dataloader("datasets/coco", 1)

dr = MyData(det_data_loader)
int8_model_path = models_dir / f"{DET_MODEL_NAME}_int8_ORT.onnx"
quantize_static(
    det_model_path,
    int8_model_path,
    dr,
    quant_format=QuantFormat.QDQ,
)