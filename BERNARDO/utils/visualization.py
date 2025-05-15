import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from BERNARDO.utils.data_utils import class_names


def visualize_predictions(model, X_test, y_test, num_samples=15):
    """Visualize model predictions on random test samples"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    # Select random samples
    indices = np.random.choice(len(y_test), num_samples, replace=False)
    X_samples = X_test[indices]
    y_samples = y_test[indices]
    
    # Convert to tensors
    X_tensor = torch.tensor(X_samples, dtype=torch.float32).to(device)
    
    # Get predictions
    with torch.no_grad():
        outputs = model(X_tensor)
        _, preds = outputs.max(1)
        probs = F.softmax(outputs, dim=1)
        
    preds = preds.cpu().numpy()
    probs = probs.cpu().numpy()
    
    # Plot samples with predictions
    rows = int(np.ceil(num_samples / 5))
    plt.figure(figsize=(15, rows * 3))
    
    for i in range(num_samples):
        plt.subplot(rows, 5, i + 1)
        plt.imshow(X_samples[i][0], cmap='gray')
        
        true_label = class_names[y_samples[i]]
        pred_label = class_names[preds[i]]
        pred_prob = probs[i][preds[i]] * 100
        
        # Set title color based on correct/incorrect prediction
        title_color = 'green' if preds[i] == y_samples[i] else 'red'
        plt.title(f"True: {true_label}\nPred: {pred_label}\nConf: {pred_prob:.1f}%", 
                  color=title_color, fontsize=9)
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('prediction_samples.png')
    plt.show()
    
def plot_confusion_matrix(y_true, y_pred):
    """Plot confusion matrix"""
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.show()
    
    return cm

def plot_training_curves(train_losses, val_losses, train_accs, val_accs):
    """Plot training and validation curves"""
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Accuracy')
    plt.plot(val_accs, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_curves.png')
    plt.show()