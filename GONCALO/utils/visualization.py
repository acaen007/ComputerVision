import numpy as np
import matplotlib.pyplot as plt
import torch
import os
import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

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


def plot_confusion_matrix(model, dataloader, class_names, device="cpu", save_path=None, max_batches=None):
    """
    Generates and displays a confusion matrix, optionally saving it.
    If max_batches is set, only that number of batches will be used (for speed/debug).
    """
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        total_batches = len(dataloader)
        effective_batches = min(total_batches, max_batches or total_batches)
        for batch_idx, (inputs, labels) in enumerate(dataloader):
            if max_batches is not None and batch_idx >= max_batches:
                break

            percent = (batch_idx + 1) / effective_batches * 100
            print(f"\rProcessing batch {batch_idx + 1}/{effective_batches} ({percent:.1f}%)", end="", flush=True)

            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        print("\nFinished processing selected batches.")


    cm = confusion_matrix(all_labels, all_preds)

    plt.figure(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(xticks_rotation=45, cmap="Blues")
    plt.title("Confusion Matrix")

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")
        plt.close()
    else:
        plt.show()

def generate_classification_report(model, dataloader, class_names, device="cpu", save_dir=".", max_batches=None):
    """
    Computes test loss, accuracy, classification report, and saves it as .txt and .png.
    """
    model.eval()
    criterion = torch.nn.CrossEntropyLoss()

    all_preds = []
    all_labels = []
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        total_batches = len(dataloader)
        for batch_idx, (inputs, labels) in enumerate(dataloader):
            if max_batches is not None and batch_idx >= max_batches:
                break

            percent = (batch_idx + 1) / min(total_batches, max_batches or total_batches) * 100
            print(f"\rProcessing batch {batch_idx + 1}/{min(total_batches, max_batches or total_batches)} ({percent:.1f}%)", end="", flush=True)
            
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * inputs.size(0)

            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)

        print("\nFinished processing selected batches.")

    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples * 100

    # classification report as dictionary
    report_dict = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True)
    report_str = classification_report(all_labels, all_preds, target_names=class_names)

    # save it
    txt_path = os.path.join(save_dir, "classification_report.txt")
    with open(txt_path, "w") as f:
        f.write(f"Test Loss: {avg_loss:.4f} - Test Accuracy: {accuracy:.2f}%\n\n")
        f.write("Classification Report:\n")
        f.write(report_str)
    print(f"Classification report saved as text to: {txt_path}")

    # save as an image
    df_report = pd.DataFrame(report_dict).transpose().round(4)
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.axis('off')
    table = ax.table(cellText=df_report.values,
                     colLabels=df_report.columns,
                     rowLabels=df_report.index,
                     loc='center',
                     cellLoc='center')
    table.scale(1.2, 1.2)
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    ax.set_title(f"Test Loss: {avg_loss:.4f} - Test Accuracy: {accuracy:.2f}%", fontsize=12, pad=20)
    
    img_path = os.path.join(save_dir, "classification_report.png")
    plt.savefig(img_path, bbox_inches='tight')
    plt.close()
    print(f"Classification report table saved to: {img_path}")