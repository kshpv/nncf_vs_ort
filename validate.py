from argparse import ArgumentParser
from typing import Dict, List

import numpy as np
import onnxruntime
import torch
from sklearn.metrics import accuracy_score
from torchvision import datasets
from torchvision import transforms
from tqdm import tqdm


def validate(path_to_model: str, data_loader: torch.utils.data.DataLoader,
             providers: List[str], provider_options: Dict[str, str]) -> float:
    sess = onnxruntime.InferenceSession(path_to_model, providers=providers,
                                        provider_options=provider_options)
    _input_name = sess.get_inputs()[0].name
    _output_names = [sess.get_outputs()[0].name]

    predictions = []
    references = []
    for images, target in tqdm(data_loader):
        pred = sess.run(_output_names, {_input_name: images.numpy()})[0]
        predictions.append(np.argmax(pred, axis=1))
        references.append(target)
        
    predictions = np.concatenate(predictions, axis=0)
    references = np.concatenate(references, axis=0)
    return accuracy_score(predictions, references)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('model_path', type=str)
    parser.add_argument('dataset_path', type=str, default='/home/akash/intel/datasets/imagenet/val')
    parser.add_argument('--ovep', action='store_true')
    args = parser.parse_args()
    dataset_path = args.dataset_path 
    model_path = args.model_path
    is_ovep = args.ovep

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    val_dataset = datasets.ImageFolder(
        root=dataset_path,
        transform=transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ]
        ),
    )
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False)
    providers = ['CPUExecutionProvider']
    if is_ovep:
        providers = ['OpenVINOExecutionProvider']
    top1 = validate(model_path, val_loader,
                    providers = providers,
                    provider_options = [{'device_type' : 'CPU_FP32'}])

    print(f"Accuracy @ top1: {top1:.3f}")