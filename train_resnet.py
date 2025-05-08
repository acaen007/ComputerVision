import os
import time
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from models.resnet import ResNet18
from utils.data_utils import load_data, class_names
from utils.visualization import plot_training_curves

def parse_args():
    parser = argparse.ArgumentParser(description="Train ResNet-18 on Fashion MNIST")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs to train")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--checkpoint_dir", default="checkpoints", help="Directory to save checkpoints")
    return parser.parse_args()

def train_model(model, train_loader, val_loader, epochs=10, lr=0.001, checkpoint_dir='checkpoints'):
    """Train the model with validation"""
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Training on: {device}")
    model = model.to(device)

    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5, verbose=True)
    
    # Create checkpoint directory if it doesn't exist
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Training metrics
    best_val_loss = float('inf')
    best_val_acc = 0.0
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    total_train_time = 0
    
    print(f"Starting training for {epochs} epochs...")
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        correct_train = 0
        total_train = 0
        start_time = time.time()
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            
            # Compute metrics
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total_train += targets.size(0)
            correct_train += predicted.eq(targets).sum().item()
            
            # Print progress
            if (batch_idx + 1) % 50 == 0 or (batch_idx + 1) == len(train_loader):
                print(f"Epoch {epoch+1}/{epochs} - Batch {batch_idx+1}/{len(train_loader)} - "
                      f"Loss: {train_loss/(batch_idx+1):.4f} - "
                      f"Acc: {100.*correct_train/total_train:.2f}%")
        
        epoch_time = time.time() - start_time
        total_train_time += epoch_time
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total_val += targets.size(0)
                correct_val += predicted.eq(targets).sum().item()
        
        # Compute average metrics
        train_loss = train_loss / len(train_loader)
        val_loss = val_loss / len(val_loader)
        train_acc = 100. * correct_train / total_train
        val_acc = 100. * correct_val / total_val
        
        # Store metrics
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Print epoch summary
        print(f"Epoch {epoch+1}/{epochs} completed in {epoch_time:.2f}s - "
              f"Train Loss: {train_loss:.4f} - Train Acc: {train_acc:.2f}% - "
              f"Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            print(f"Validation accuracy improved from {best_val_acc:.2f}% to {val_acc:.2f}% - Saving model...")
            best_val_acc = val_acc
            torch.save(model.state_dict(), f"{checkpoint_dir}/resnet18_fashion_mnist_best.pth")
    
    # Save final model
    torch.save(model.state_dict(), f"{checkpoint_dir}/resnet18_fashion_mnist_final.pth")
    
    print(f"Training completed in {total_train_time:.2f}s!")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    
    # Plot training curves
    plot_training_curves(train_losses, val_losses, train_accs, val_accs)
    
    return model, best_val_acc

def main():
    args = parse_args()
    
    # Load data
    train_loader, val_loader, _, _, _ = load_data(batch_size=args.batch_size)
    
    # Create model
    model = ResNet18(num_classes=10)
    
    # Print model summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: ResNet-18")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Train model
    print("\n=== Starting Model Training ===")
    model, best_val_acc = train_model(
        model, 
        train_loader, 
        val_loader, 
        epochs=args.epochs, 
        lr=args.lr,
        checkpoint_dir=args.checkpoint_dir
    )
    
    print("\n=== Training Completed ===")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    print(f"Model checkpoints saved to '{args.checkpoint_dir}' directory")
    print(f"Training curves saved to 'training_curves.png'")

if __name__ == "__main__":
    main()