
# Installation steps:
1) Install NNCF - https://github.com/openvinotoolkit/nncf 
2) pip install -r requirements.txt


# Run quantization

For quantization the 300 samples from ImageNet is used.

## ONNX model

1) python nncf_quantize.py

The quantized model will be put in models dir with the name 'resnet50-v1-12_int8_NNCF.onnx'

2) python ort_quantize.py

The quantized model will be put in models dir with the name 'resnet50-v1-12_int8_ORT.onnx'

## OpenVINO IR model

1) Convert model from ONNX to IR by:
    cd models
    mo -m resnet50-v1-12.onnx

2) Run quantization:
    python ov_nncf_quantize.py

# Run validation

1) Download ImageNet dataset for validation
2) python validate.py <model_path> <dataset_path> --ovep (if run OV EP)

Results:
Resnet50 opset 12:
FP32 (ONNXRuntime with CPUExecutionProvider) - 
OpenVINO NNCF int8 (OpenVINO inference) - 0.739660
NNCF int8 (ONNXRuntime with OpenVINOExecutionProvider) - 
ONNXRuntime quantization int8 (ONNXRuntime with CPUExecutionProvider) - 0.729
Resnet50 opset 13:
FP32 (ONNXRuntime with CPUExecutionProvider) - 
OpenVINO NNCF int8 (OpenVINO inference) - 0.739660
NNCF int8 (ONNXRuntime with OpenVINOExecutionProvider) - 0.740240
ONNXRuntime quantization int8 (ONNXRuntime with CPUExecutionProvider) - 

