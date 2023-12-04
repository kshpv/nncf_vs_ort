

# Installation steps:
1) Install NNCF - https://github.com/openvinotoolkit/nncf 
2) ```pip install -r requirements.txt```

# Run quantization

For quantization the 300 samples from MSCOCO is used.
1) ```python download_dataset.py```
2) ```python get_model.py```

## ONNX model

1) ```python nncf_quantize.py```

The quantized model will be put in models dir with the name 'resnet50-v1-12_int8_NNCF.onnx'

2) ```python ort_quantize.py```

The quantized model will be put in models dir with the name 'resnet50-v1-12_int8_ORT.onnx'

## OpenVINO IR model

1) Run quantization:
    ```python ov_nncf_quantize.py```

# Run validation

1) Download ImageNet dataset for validation
2) ```python validate.py <model_path> --ovep (if run OV EP)```