# Computer Vision Model Inference Pipeline (DETR, ResNet, VGG16)

## Requirements

To run this project, you'll need the following Python packages:

```bash
pip install -r requirements.txt
```

## Project Description
This project contains an interactive pipeline to run pre-trained deep learning models for image classification using the Fashion MNIST dataset.
The user can choose between different models implemented by team members:

## DETR (transformer-based object detector)

## ResNet-18

## VGG16

Each model is organized under its own folder and implements its own inference logic.
The pipeline handles user input, directory setup, and execution automatically.

## Folder Structure
```bash
ANDY/                          # DETR model implementation
├── inference/                 # Main inference logic
├── models/                    # Model architecture
├── utils/                     # Preprocessing and visualization utilities
BERNARDO/                     # ResNet model implementation
├── inference/                 # Main inference logic
├── models/                    # Model architecture
├── utils/                     # Preprocessing and visualization utilities
GONCALO/                       # VGG16 model implementation
├── inference/                 # Main inference logic
├── models/                    # Model architecture
├── utils/                     # Preprocessing and visualization utilities
main.py                        # Central script to launch any model
requirements.txt
output/                        # Output visualizations saved here
```
## How to Run

1. Clone the repository
```bash
git clone https://github.com/acaen007/ComputerVision.git
cd ComputerVision
```
2. Run the main script
```bash
python main.py or python3 main.py
```
You will be asked which model to run:
```bash
Which model would you like to run? (DETR, RESNET, VGG16, or type 'exit' to quit):
```
Choose your model, and the inference script for that model will run.

# VGG16 Model Details (GONCALO)
-Automatically downloads pretrained weights from Hugging Face:
https://huggingface.co/thearezes/vgg16-fashionmnist

-Runs inference on the Fashion MNIST test set

-Saves a grid of predicted images to:
```bash
output/VGG16/vgg16_predictions.png
```
## Example Output
Once VGG16 finishes running, the output file vgg16_predictions.png will be generated with model predictions and confidences for sample images.

# Authors

Bernardo – ResNet

Andy – DETR

Gonçalo Arezes – VGG16

## Notes
Make sure all folders (inference, models, utils) contain __init__.py files so Python can import across directories.

For any issues, please contact one of the authors.
