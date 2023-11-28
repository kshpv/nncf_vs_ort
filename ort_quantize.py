from argparse import ArgumentParser

import resnet50_data_reader
from onnxruntime.quantization import QuantFormat
from onnxruntime.quantization import quantize_static

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--model_path', type=str, default='./resnet50-v1-12.onnx')
    parser.add_argument('--output_path', type=str, default='./resnet50-v1-12_int8_ORT.onnx')
    parser.add_argument('--dataset_path', type=str, default='./imagenet_300_samples')
    args = parser.parse_args()
    
    model_path = args.model_path
    int8_model_path = args.output_path
    dataset_path = args.dataset_path
    
    dr = resnet50_data_reader.ResNet50DataReader(dataset_path, model_path)
    quantize_static(
        model_path,
        int8_model_path,
        dr,
        quant_format=QuantFormat.QDQ,
    )
    print("Calibrated and quantized model saved.")
