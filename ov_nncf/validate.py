from argparse import ArgumentParser

import numpy as np
import openvino as ov
import torch
from sklearn.metrics import accuracy_score
from torchvision import datasets
from torchvision import transforms
from tqdm import tqdm


def validate(model: ov.Model, val_loader: torch.utils.data.DataLoader) -> float:
    predictions = []
    references = []

    compiled_model = ov.compile_model(model)
    output = compiled_model.outputs[0]

    for images, target in tqdm(val_loader):
        pred = compiled_model(images)[output]
        predictions.append(np.argmax(pred, axis=1))
        references.append(target)

    predictions = np.concatenate(predictions, axis=0)
    references = np.concatenate(references, axis=0)
    return accuracy_score(predictions, references)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('model_path', type=str)
    parser.add_argument('dataset_path', type=str, default='/home/akash/intel/datasets/imagenet/val')
    
    args = parser.parse_args()
    dataset_path = args.dataset_path 
    model_path = args.model_path
    
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

    top1 = validate(model_path, val_loader)

    print(f"Accuracy @ top1: {top1:.3f}")