import os
import sys

import torch
from huggingface_hub import hf_hub_download
from GONCALO.models.vgg16_model import build_vgg16_model
from GONCALO.utils.preprocessing import load_fashionmnist
from GONCALO.utils.visualization import (visualize_model_predictions, plot_dataset_samples)

def run_vgg16_inference(project_root=None):
    """Runs VGG16 inference pipeline on FashionMNIST."""
    device = torch.device("cpu")

    print("Loading FashionMNIST dataset...")
    _, testloader, class_names = load_fashionmnist(batch_size=32)

    print("Loading pretrained VGG16 model...")
    model = build_vgg16_model(device=device)

    model_path = hf_hub_download(repo_id="thearezes/vgg16-fashionmnist", filename="vgg16_fashionmnist.pth")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)

    print("Showing dataset samples...")
    plot_dataset_samples(testloader, class_names)
    output_path = os.path.join(project_root or ".", "output", "VGG16", "vgg16_predictions.png")


    print("Running model predictions...")
    visualize_model_predictions(model, testloader, class_names, device=device, num_images=16, save_path=output_path)
    print("VGG16 inference completed.")

if __name__ == "__main__":
    run_vgg16_inference()