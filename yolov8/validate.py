from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import onnxruntime
import torch
from tqdm.notebook import tqdm
from ultralytics import YOLO
from ultralytics.yolo.cfg import get_cfg
from ultralytics.yolo.data.utils import check_det_dataset
from ultralytics.yolo.utils import DEFAULT_CFG
from ultralytics.yolo.utils import ops
from ultralytics.yolo.utils.metrics import ConfusionMatrix


def test(model_path:str, data_loader:torch.utils.data.DataLoader, validator, num_samples:int = None, providers = None, provider_options = None):
    """
    OpenVINO YOLOv8 model accuracy validation function. Runs model validation on dataset and returns metrics
    Parameters:
        model (Model): OpenVINO model
        data_loader (torch.utils.data.DataLoader): dataset loader
        validator: instance of validator class
        num_samples (int, *optional*, None): validate model only on specified number samples, if provided
    Returns:
        stats: (Dict[str, float]) - dictionary with aggregated accuracy metrics statistics, key is metric name, value is metric value
    """
    validator.seen = 0
    validator.jdict = []
    validator.stats = []
    validator.batch_i = 1
    validator.confusion_matrix = ConfusionMatrix(nc=validator.nc)
    sess = onnxruntime.InferenceSession(model_path, providers=providers,
                                        provider_options=provider_options)
    _input_name = sess.get_inputs()[0].name
    _output_names = [sess.get_outputs()[0].name]
    for batch_i, batch in enumerate(tqdm(data_loader, total=num_samples)):
        if num_samples is not None and batch_i == num_samples:
            break
        batch = validator.preprocess(batch)
        results = sess.run(_output_names, {_input_name: batch["img"].cpu().detach().numpy()})[0]
        preds = torch.from_numpy(results)
        preds = validator.postprocess(preds)
        validator.update_metrics(preds, batch)
    stats = validator.get_stats()
    return stats

def test_ov(model, core, data_loader:torch.utils.data.DataLoader, validator, num_samples:int = None):
    """
    OpenVINO YOLOv8 model accuracy validation function. Runs model validation on dataset and returns metrics
    Parameters:
        model (Model): OpenVINO model
        data_loader (torch.utils.data.DataLoader): dataset loader
        validator: instance of validator class
        num_samples (int, *optional*, None): validate model only on specified number samples, if provided
    Returns:
        stats: (Dict[str, float]) - dictionary with aggregated accuracy metrics statistics, key is metric name, value is metric value
    """
    validator.seen = 0
    validator.jdict = []
    validator.stats = []
    validator.batch_i = 1
    validator.confusion_matrix = ConfusionMatrix(nc=validator.nc)
    model.reshape({0: [1, 3, -1, -1]})
    compiled_model = core.compile_model(model)
    for batch_i, batch in enumerate(tqdm(data_loader, total=num_samples)):
        if num_samples is not None and batch_i == num_samples:
            break
        batch = validator.preprocess(batch)
        results = compiled_model(batch["img"])
        preds = torch.from_numpy(results[compiled_model.output(0)])
        preds = validator.postprocess(preds)
        validator.update_metrics(preds, batch)
    stats = validator.get_stats()
    return stats


def print_stats(stats:np.ndarray, total_images:int, total_objects:int):
    """
    Helper function for printing accuracy statistic
    Parameters:
        stats: (Dict[str, float]) - dictionary with aggregated accuracy metrics statistics, key is metric name, value is metric value
        total_images (int) -  number of evaluated images
        total objects (int)
    Returns:
        None
    """
    print("Boxes:")
    mp, mr, map50, mean_ap = stats['metrics/precision(B)'], stats['metrics/recall(B)'], stats['metrics/mAP50(B)'], stats['metrics/mAP50-95(B)']
    # Print results
    s = ('%20s' + '%12s' * 6) % ('Class', 'Images', 'Labels', 'Precision', 'Recall', 'mAP@.5', 'mAP@.5:.95')
    print(s)
    pf = '%20s' + '%12i' * 2 + '%12.3g' * 4  # print format
    print(pf % ('all', total_images, total_objects, mp, mr, map50, mean_ap))
    if 'metrics/precision(M)' in stats:
        s_mp, s_mr, s_map50, s_mean_ap = stats['metrics/precision(M)'], stats['metrics/recall(M)'], stats['metrics/mAP50(M)'], stats['metrics/mAP50-95(M)']
        # Print results
        s = ('%20s' + '%12s' * 6) % ('Class', 'Images', 'Labels', 'Precision', 'Recall', 'mAP@.5', 'mAP@.5:.95')
        print(s)
        pf = '%20s' + '%12i' * 2 + '%12.3g' * 4  # print format
        print(pf % ('all', total_images, total_objects, s_mp, s_mr, s_map50, s_mean_ap))
    
def is_openvino_model(model_path: str):
    if '.xml' in Path(model_path).suffix:
        return True
    return False
    
if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('model_path', type=str)
    parser.add_argument('--ovep', action='store_true')
    terminal_args = parser.parse_args()
    
    is_ov = is_openvino_model(terminal_args.model_path)

    DET_MODEL_NAME = "yolov8n"
    models_dir = Path('./models')
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

    NUM_TEST_SAMPLES = 300

    if is_ov:
        import openvino as ov
        core = ov.Core()
        det_ov_model = core.read_model(terminal_args.model_path)
        det_compiled_model = core.compile_model(det_ov_model)     
        fp_det_stats = test_ov(det_ov_model, core, det_data_loader, det_validator, num_samples=NUM_TEST_SAMPLES)   
    else:
        if terminal_args.ovep:
            print ('OpenVINOEP will be used')
            fp_det_stats = test(terminal_args.model_path, det_data_loader, det_validator, num_samples=NUM_TEST_SAMPLES, providers=['OpenVINOExecutionProvider'])
        else:
            print ('CPUEP will be used')
            fp_det_stats = test(terminal_args.model_path, det_data_loader, det_validator, num_samples=NUM_TEST_SAMPLES, providers=['CPUExecutionProvider'])
            
    print_stats(fp_det_stats, det_validator.seen, det_validator.nt_per_class.sum())