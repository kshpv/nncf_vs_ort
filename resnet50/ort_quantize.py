from argparse import ArgumentParser

import onnx
from onnxruntime.quantization import QuantFormat
from onnxruntime.quantization import quantize_static
from resnet50_data_reader import ResNet50DataReader

from nncf.onnx.graph.onnx_helper import get_edge_shape

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('model_path', nargs='?', type=str, default='./models/resnet50-v1-12.onnx')
    parser.add_argument('output_path', nargs='?', type=str, default='./models/resnet50-v1-12_int8_ORT.onnx')
    parser.add_argument('dataset_path', nargs='?', type=str, default='./imagenet_300_samples')
    args = parser.parse_args()
    
    model_path = args.model_path
    int8_model_path = args.output_path
    dataset_path = args.dataset_path
    
    model = onnx.load(model_path)
    input_shape = get_edge_shape(model.graph.input[0])
    dataset = ResNet50DataReader(dataset_path, input_shape, model.graph.input[0])
    dr = ResNet50DataReader(dataset_path, input_shape, model.graph.input[0])
    quantize_static(
        model_path,
        int8_model_path,
        dr,
        quant_format=QuantFormat.QDQ
    )
    print("Calibrated and quantized model saved.")
