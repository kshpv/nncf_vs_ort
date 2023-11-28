# Copyright (c) 2023 Intel Corporation
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from argparse import ArgumentParser

import openvino.runtime as ov
from resnet50_data_reader import ResNet50DataReader

import nncf

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--model_path', type=str, default='./resnet50-v1-12/resnet50-v1-12.xml')
    parser.add_argument('--output_path', type=str, default='./resnet50-v1-12/resnet50-v1-12_int8.xml')
    parser.add_argument('--dataset_path', type=str, default='../imagenet_300_samples')
    args = parser.parse_args()
    
    model_path = args.model_path
    int8_model_path = args.output_path
    dataset_path = args.dataset_path
    
    model = ov.Core().read_model(model_path)
    dataset = ResNet50DataReader(dataset_path, model_path)
    calibration_dataset = nncf.Dataset(dataset)
    ov_quantized_model = nncf.quantize(model, calibration_dataset)

    ov.save_model(ov_quantized_model, int8_model_path, compress_to_fp16=False)
