
from argparse import ArgumentParser

import onnx
from resnet50_data_reader import ResNet50DataReader

import nncf
from nncf.onnx.graph.onnx_helper import get_edge_shape

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('model_path', nargs='?', type=str, default='./models/resnet50-v1-12.onnx')
    parser.add_argument('output_path', nargs='?', type=str, default='./models/resnet50-v1-12_int8_NNCF.onnx')
    parser.add_argument('dataset_path', nargs='?', type=str, default='./imagenet_300_samples')
    args = parser.parse_args()
    
    model_path = args.model_path
    int8_model_path = args.output_path
    dataset_path = args.dataset_path
    
    model = onnx.load(model_path)
    input_shape = get_edge_shape(model.graph.input[0])
    dataset = ResNet50DataReader(dataset_path, input_shape, model.graph.input[0])
    calibration_dataset = nncf.Dataset(dataset)
    onnx_quantized_model = nncf.quantize(model, calibration_dataset)

    onnx.save(onnx_quantized_model, int8_model_path)
