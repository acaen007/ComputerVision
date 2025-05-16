import sys
import os
import torch
import random
import json
import numpy as np
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

from transformers import DetrForObjectDetection, DetrImageProcessor
from ANDY.utils.preprocessing import load_fashionmnist

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#load model 
processor = DetrImageProcessor.from_pretrained("acaen/detr-fashionmnist")
model = DetrForObjectDetection.from_pretrained("acaen/detr-fashionmnist").to(device)
model.eval()

#load test data
_, testloader, class_names = load_fashionmnist(batch_size=1)

# For reverse normalization
unnorm = torch.nn.Sequential(
    transforms.Normalize(mean=[0., 0., 0.], std=[1/0.229, 1/0.224, 1/0.225]),
    transforms.Normalize(mean=[-0.485, -0.456, -0.406], std=[1., 1., 1.])
)

output_dir = "output/DETR"
os.makedirs(output_dir, exist_ok=True)

# === Draw detection boxes on sample image ===
def draw_prediction(image, boxes, labels, scores, class_names, save_path):
    plt.figure(figsize=(6, 6))
    plt.imshow(image)
    ax = plt.gca()
    for box, label, score in zip(boxes, labels, scores):
        xmin, ymin, xmax, ymax = box
        width, height = xmax - xmin, ymax - ymin
        rect = patches.Rectangle((xmin, ymin), width, height, linewidth=2,
                                 edgecolor='red', facecolor='none')
        ax.add_patch(rect)
        ax.text(xmin, ymin - 5, f"{class_names[label]}: {score:.2f}", color='white',
                backgroundcolor='red', fontsize=8)
    plt.axis("off")
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1)
    plt.close()

def run_detr_inference():
    print("Running DETR inference on test set...")

    y_true = []
    y_pred = []
    samples = []

    N = 500  # decide how many images to process
    for idx, (img_tensor, label) in enumerate(testloader):
        print(f"Processing image {idx+1}/{N}...")
        if idx >= N:
            break
        img_tensor = img_tensor.squeeze(0)  # remove batch dim
        img = transforms.ToPILImage()(unnorm(img_tensor)).convert("RGB")

        inputs = processor(images=img, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        result = processor.post_process_object_detection(outputs, threshold=0.5, target_sizes=[img.size])[0]

        pred_labels = result["labels"].cpu().numpy()
        pred_scores = result["scores"].cpu().numpy()

        #for classification metrics, take top predicted label only
        pred_class = int(pred_labels[0]) if len(pred_labels) > 0 else -1
        y_pred.append(pred_class)
        y_true.append(int(label))

        #save 5 random samples for visualization
        if len(samples) < 5 and random.random() < 0.05:
            samples.append((img, result, idx))

    #save visual predictions
    for i, (img, result, idx) in enumerate(samples):
        boxes = result["boxes"].cpu().numpy()
        labels = result["labels"].cpu().numpy()
        scores = result["scores"].cpu().numpy()
        save_path = os.path.join(output_dir, f"sample_pred_{i+1}.png")
        draw_prediction(img, boxes, labels, scores, class_names, save_path)
        print(f"Saved prediction: {save_path}")

    cm = confusion_matrix(y_true, y_pred, labels=list(range(10)))
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "confusion_matrix.png"))
    plt.close()
    print("Confusion matrix saved.")

    #Classification Report
    report = classification_report(
        y_true, y_pred, labels=list(range(10)),
        target_names=class_names, output_dict=True
    )
    with open(os.path.join(output_dir, "classification_report.json"), "w") as f:
        json.dump(report, f, indent=4)
    print("Classification report saved.")

