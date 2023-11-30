
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

2) Run quntization:
    python 

Download ImageNet dataset for 