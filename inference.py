import os
import argparse
import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import classification_report, accuracy_score
from models.resnet import ResNet18
from utils.data_utils import load_data, class_names
from utils.visualization import visualize_predictions, plot_confusion_matrix

def parse_args():
    parser = argparse.ArgumentParser(description="Fashion MNIST Classification Pipeline")
    parser.add_argument("--model", default="resnet18", help="Model architecture to use")
    parser.add_argument("--checkpoint", default="checkpoints/resnet18_fashion_mnist_best.pth", 
                      help="Path to model checkpoint")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for inference")
    parser.add_argument("--num_samples", type=int, default=15, help="Number of samples to visualize")
    parser.add_argument("--save_dir", default="results", help="Directory to save results")
    return parser.parse_args()

def evaluate_model(model, test_loader, X_test, y_test):
    """Evaluate model on test set with detailed metrics"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    # Prediction arrays
    all_preds = []
    all_targets = []
    
    # Test metrics
    test_loss = 0.0
    correct = 0
    total = 0
    criterion = torch.nn.CrossEntropyLoss()
    
    print("Evaluating model on test set...")
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            # Store predictions and targets for metrics
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    
    # Compute average loss and accuracy
    test_loss = test_loss / len(test_loader)
    test_acc = 100. * correct / total
    
    print(f"Test Loss: {test_loss:.4f} - Test Accuracy: {test_acc:.2f}%")
    
    # Generate detailed classification report
    report = classification_report(all_targets, all_preds, target_names=class_names, digits=4)
    print("\nClassification Report:")
    print(report)
    
    # Plot confusion matrix
    cm = plot_confusion_matrix(all_targets, all_preds)
    
    return test_acc, all_preds, all_targets

def main():
    args = parse_args()
    
    # Create save directory if it doesn't exist
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Load data
    _, _, test_loader, X_test, y_test = load_data(batch_size=args.batch_size)
    
    # Initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    if args.model == "resnet18":
        model = ResNet18(num_classes=10)
    else:
        raise ValueError(f"Unsupported model: {args.model}")
    
    # Load checkpoint
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.to(device)
    model.eval()
    
    # Run inference and evaluate
    print("\n=== Running Inference ===")
    test_acc, predictions, targets = evaluate_model(model, test_loader, X_test, y_test)
    
    # Visualize sample predictions
    print("\n=== Visualizing Sample Predictions ===")
    visualize_predictions(model, X_test, y_test, num_samples=args.num_samples)
    
    # Save evaluation results
    results_file = f"{args.save_dir}/evaluation_results.txt"
    with open(results_file, 'w') as f:
        f.write(f"Fashion MNIST ResNet-18 Evaluation Results\n\n")
        f.write(f"Test Accuracy: {test_acc:.4f}%\n\n")
        f.write("Classification Report:\n")
        f.write(classification_report(targets, predictions, target_names=class_names, digits=4))
    
    print(f"\nEvaluation results saved to {results_file}")
    print(f"Confusion matrix saved to 'confusion_matrix.png'")
    print(f"Sample predictions saved to 'prediction_samples.png'")

if __name__ == "__main__":
    main()