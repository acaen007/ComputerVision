import torch
import numpy as np
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

# Class names for Fashion MNIST
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

def load_data(batch_size=128):
    """Load and prepare Fashion MNIST dataset"""
    print("Loading Fashion MNIST dataset...")
    
    # Define transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    # Load the dataset
    train_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
    
    # Split into train and validation sets (80% train, 20% validation)
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    # Prepare X_test and y_test arrays for visualization
    X_test = []
    y_test = []
    
    for images, labels in test_loader:
        X_test.append(images.numpy())
        y_test.append(labels.numpy())
    
    X_test = np.concatenate(X_test, axis=0)
    y_test = np.concatenate(y_test, axis=0)
    
    print(f"Dataset loaded: {train_size} training samples, {val_size} validation samples, {len(test_dataset)} test samples")
    
    return train_loader, val_loader, test_loader, X_test, y_test