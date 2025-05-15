import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def imshow(img, title=None):
    """
    Unnormalizes and displays a single image.
    """
    img = img.cpu().numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = std * img + mean
    img = np.clip(img, 0, 1)
    plt.imshow(img)
    if title:
        plt.title(title)
    plt.pause(0.001)


def plot_dataset_samples(dataloader, class_names):
    """
    Plots a few sample images from the dataset.
    """
    dataiter = iter(dataloader)
    images, labels = next(dataiter)

    plt.figure(figsize=(10, 10))
    for i in range(9):
        plt.subplot(3, 3, i + 1)
        imshow(images[i], title=class_names[labels[i]])
    plt.tight_layout()
    plt.show()


def plot_training_curves(history):
    """
    Plots training and validation loss and accuracy over epochs.
    """
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train')
    plt.plot(history['val_loss'], label='Validation')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train')
    plt.plot(history['val_acc'], label='Validation')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()

    plt.tight_layout()
    plt.show()


def visualize_model_predictions(model, dataloader, class_names, device="cpu", num_images=25, save_path=None):
    """
    Displays a grid of model predictions with confidence and correctness coloring.
    """
    model.eval()
    images_shown = 0
    plt.figure(figsize=(12, 12))

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            confidences, _ = torch.max(probs, 1)

            for i in range(inputs.size(0)):
                if images_shown >= num_images:
                    plt.tight_layout()
                    if save_path:
                        plt.savefig(save_path)
                        print(f"Saved prediction plot to {save_path}")
                    else:
                        plt.show()
                    return

                ax = plt.subplot(int(np.sqrt(num_images)), int(np.sqrt(num_images)), images_shown + 1)
                inp = inputs[i].cpu().numpy().transpose((1, 2, 0))
                inp = np.clip(inp * 0.229 + 0.485, 0, 1)
                ax.imshow(inp)
                pred_label = class_names[preds[i]]
                true_label = class_names[labels[i]]
                confidence = confidences[i].item() * 100

                color = 'green' if pred_label == true_label else 'red'
                ax.set_title(f'True: {true_label}\nPred: {pred_label}\nConf: {confidence:.1f}%', color=color, fontsize=8)
                ax.axis('off')
                images_shown += 1


def plot_confusion_matrix(model, dataloader, class_names, device="cpu"):
    """
    Generates and displays a confusion matrix.
    """
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(xticks_rotation=45, cmap="Blues")
    plt.title("Confusion Matrix")
    plt.show()
