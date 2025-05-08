# Fashion MNIST Classification with ResNet-18

## Requirements

To run this project, you'll need the following Python packages:

```bash
pip install torch torchvision numpy matplotlib scikit-learn seaborn
```

## Dataset

This project uses the **Fashion MNIST** dataset, which contains 70,000 grayscale images (28x28 pixels) of 10 fashion categories.  
The dataset will be automatically downloaded when running the scripts.

## File Structure

```
models/resnet.py             # Implements the ResNet-18 model architecture with residual blocks
utils/data_utils.py          # Handles dataset loading, preprocessing, and train/validation/test split
utils/visualization.py       # Functions for visualizing model predictions and performance metrics
train.py                     # Script for training deep learning models on Fashion MNIST
inference.py                 # Script for evaluating trained models and visualizing predictions
```

## How to Run

### Training

To train any model (including ResNet-18), run:

```bash
python train.py --model [model_name] --epochs [num_epochs] --batch_size [batch_size] --lr [learning_rate] --checkpoint_dir [save_directory]
```

**Example:**

```bash
python train.py --model resnet18 --epochs 10 --batch_size 128 --lr 0.001 --checkpoint_dir checkpoints
```

### Inference

To evaluate a trained model and visualize predictions:

```bash
python inference.py --model [model_name] --checkpoint [checkpoint_path] --batch_size [batch_size] --num_samples [visualization_samples] --save_dir [results_directory]
```

**Example:**

```bash
python inference.py --model resnet18 --checkpoint checkpoints/resnet18_fashion_mnist_best.pth --num_samples 15 --save_dir results
```
