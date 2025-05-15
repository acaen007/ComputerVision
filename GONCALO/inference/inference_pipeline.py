import sys
import os
import torch
from huggingface_hub import hf_hub_download
from models.vgg16_model import build_vgg16_model
from utils.preprocessing import load_fashionmnist
from utils.visualization import (visualize_model_predictions, plot_dataset_samples)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

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

    print("Running model predictions...")
    visualize_model_predictions(model, testloader, class_names, device=device, num_images=20)

    print("VGG16 inference completed.")

if __name__ == "__main__":
    run_vgg16_inference()
