import sys
import os
# Allow imports from parent directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import torch
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from transformers import DetrForObjectDetection, DetrImageProcessor
from utils.preprocessing import load_fashionmnist



device = torch.device("cpu")
model_dir = "./ANDY/detr_fashionmnist"

# Load processor and model
processor = DetrImageProcessor.from_pretrained(model_dir, local_files_only=True)
model = DetrForObjectDetection.from_pretrained(model_dir, local_files_only=True).to(device)

# Load test images
_, testloader, class_names = load_fashionmnist(batch_size=5)
image_batch, _ = next(iter(testloader))  # (5, 3, 224, 224)

# Define unnormalization
unnorm = torch.nn.Sequential(
    transforms.Normalize(mean=[0., 0., 0.], std=[1/0.229, 1/0.224, 1/0.225]),
    transforms.Normalize(mean=[-0.485, -0.456, -0.406], std=[1., 1., 1.])
)

# === Visualization ===
def draw_and_save_detr_results(image, boxes, labels, scores, class_names, save_path):
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
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1)
    print(f"Saved: {save_path}")
    plt.close()

# Run inference on each image

def run_detr_inference():
    for idx in range(5):
        img_tensor = image_batch[idx]
        img = transforms.ToPILImage()(unnorm(img_tensor))

        inputs = processor(images=img, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        results = processor.post_process_object_detection(outputs, threshold=0.5, target_sizes=[img.size])[0]

        boxes = results["boxes"].cpu().numpy()
        labels = results["labels"].cpu().numpy()
        scores = results["scores"].cpu().numpy()

        save_path = f"output/detr_result_{idx+1}.png"
        draw_and_save_detr_results(img, boxes, labels, scores, class_names, save_path)
