import time
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import onnx
import onnxruntime


def gen_data(input_shape, runs):
    input_data = []
    for i in range(runs):
        input_data.append(np.random.random(input_shape).astype(np.float32))
    return input_data

def benchmark(model_path, input_shape, providers=None):
    session = onnxruntime.InferenceSession(model_path, providers=providers)
    input_name = session.get_inputs()[0].name

    total = 0.0
    runs = 1000
    input_data = gen_data(input_shape, runs)
    
    # Warming up
    _ = session.run([], {input_name: input_data[0]})
    for i in range(runs):
        start = time.perf_counter()
        _ = session.run([], {input_name: input_data[i]})
        end = (time.perf_counter() - start) * 1000
        total += end
    total /= runs
    print(f"Avg: {total:.2f}ms")


def is_openvino_model(model_path: str):
    if '.xml' in Path(model_path).suffix:
        return True
    return False
    
def becnhmark_ov(model_path):
    import openvino as ov
    compiled_model = ov.compile_model(model_path)
    input_shape = [1, 3 , 224, 224]
    runs = 100

    input_data = gen_data(input_shape, runs)
    total = 0.0
    for i in range(runs):
        start = time.perf_counter()
        _ = compiled_model(input_data[i])
        end = (time.perf_counter() - start) * 1000
        total += end
    total /= runs
    print(f"Avg: {total:.2f}ms")


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('model_path', type=str)
    parser.add_argument('--ovep', action='store_true')
    args = parser.parse_args()
    model_path = args.model_path
    is_ov_ep = args.ovep
    if is_openvino_model(model_path):
        becnhmark_ov(model_path)
    else:
        model = onnx.load(args.model_path)
        input_shape = [d.dim_value for d in model.graph.input[0].type.tensor_type.shape.dim]
        input_shape[0] = 1
        providers = ['CPUExecutionProvider']
        if is_ov_ep:
            providers = ['OpenVINOExecutionProvider']
        print (f'The {providers[0]} will be used')
        benchmark(args.model_path, input_shape, providers)