import torch
from models.vgg16_model import build_vgg16_model
from utils.preprocessing import load_fashionmnist
from utils.visualization import visualize_model_predictions

def main():
    device = torch.device("cpu")  # Required: inference must run on CPU

    # Load test data and class names using exact preprocessing
    _, testloader, class_names = load_fashionmnist(batch_size=32)

    # Load VGG16 model and weights
    model = build_vgg16_model(device=device)
    model.load_state_dict(torch.load("models/vgg16_fashionmnist.pth", map_location=device))
    model.to(device)

    # Run predictions and visualize
    visualize_model_predictions(model, testloader, class_names, device=device, num_images=25)


if __name__ == "__main__":
    main()
