import time
from argparse import ArgumentParser

import numpy as np
import onnx
import onnxruntime


def benchmark(model_path, input_shape, providers=None):
    session = onnxruntime.InferenceSession(model_path, providers=providers)
    input_name = session.get_inputs()[0].name

    total = 0.0
    runs = 1000
    input_data = np.zeros(input_shape, np.float32)
    # Warming up
    _ = session.run([], {input_name: input_data})
    for i in range(runs):
        start = time.perf_counter()
        _ = session.run([], {input_name: input_data})
        end = (time.perf_counter() - start) * 1000
        total += end
    total /= runs
    print(f"Avg: {total:.2f}ms")
    
if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('model_path', type=str)
    args = parser.parse_args()
    model = onnx.load(args.model_path)
    input_shape = [d.dim_value for d in model.graph.input[0].type.tensor_type.shape.dim]
    input_shape[0] = 1
    benchmark(args.model_path, input_shape)