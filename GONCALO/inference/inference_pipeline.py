import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from huggingface_hub import hf_hub_download
from models.vgg16_model import build_vgg16_model
from utils.preprocessing import load_fashionmnist
from utils.visualization import visualize_model_predictions,plot_dataset_samples

def main():
    device = torch.device("cpu")  # Required: inference must run on CPU

    print("Loading FashionMNIST dataset...")
    _, testloader, class_names = load_fashionmnist(batch_size=32)

    print("Loading pre-trained VGG16 model...")
    model = build_vgg16_model(device=device)

    # download model weights from Hugging Face Hub
    model_path = hf_hub_download(
        repo_id="thearezes/vgg16-fashionmnist",  # Make sure this matches repo
        filename="vgg16_fashionmnist.pth"
    )

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)

    print("Showing dataset samples...")
    plot_dataset_samples(testloader, class_names)

    print("Running inference and visualizing predictions...")
    visualize_model_predictions(model, testloader, class_names, device=device, num_images=25)


if __name__ == "__main__":
    main()
