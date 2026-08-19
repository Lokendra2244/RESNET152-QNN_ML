import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


def invert_labels(target):
    return 1 if target == 0 else 0


# ==========================================
# ⚠️ THE EXACT ARCHITECTURE FROM YOUR TRAINING SCRIPT ⚠️
# ==========================================


class ClassicalResNet(nn.Module):
    def __init__(self, unfreeze_from: str = "layer3"):
        super().__init__()
        resnet = models.resnet152(weights="DEFAULT")
        unfreeze = False
        for name, child in resnet.named_children():
            if name == unfreeze_from:
                unfreeze = True
            for param in child.parameters():
                param.requires_grad = unfreeze
        # This creates keys like 'feature_extractor.0.weight' which matches your saved .pth file
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])
        self.dropout = nn.Dropout(p=0.5)
        self.classifier = nn.Linear(2048, 2)  # The REAL 2048 -> 2 mapping

    def forward(self, x):
        x = self.feature_extractor(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.classifier(x)
        return x


if __name__ == "__main__":
    # ==========================================
    # 1. Configuration & Hyperparameters
    # ==========================================
    TEST_DATA_DIR = "./cbis_ddsm/test"
    MODEL_WEIGHTS_PATH = "./classical_resnet_weights_best.pth"
    BATCH_SIZE = 32

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ==========================================
    # 2. Data Loading
    # ==========================================
    test_transforms = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    print("Loading test dataset...")

    test_dataset = datasets.ImageFolder(
        root=TEST_DATA_DIR, transform=test_transforms, target_transform=invert_labels
    )

    test_dataset.classes = ["MALIGNANT", "BENIGN"]
    class_names = test_dataset.classes

    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2
    )

    print(f"Total test images: {len(test_dataset)}")
    print(f"Classes aligned to model output: {class_names}")

    # ==========================================
    # 3. Model Initialization
    # ==========================================
    print("Initializing EXACT Classical ResNet152 baseline...")

    # Initialize your specific class
    model = ClassicalResNet()

    try:
        state_dict = torch.load(MODEL_WEIGHTS_PATH, map_location=device)
        # We can use strict=True now because the architecture matches the save file flawlessly!
        model.load_state_dict(state_dict, strict=True)
        print(f"✅ Successfully loaded weights from {MODEL_WEIGHTS_PATH}")
    except Exception as e:
        print(f"❌ Error loading weights: {e}")
        exit()

    model = model.to(device)
    model.eval()

    # ==========================================
    # 4. Inference Loop
    # ==========================================
    all_preds = []
    all_labels = []
    running_loss = 0.0

    criterion = nn.CrossEntropyLoss()

    print("\nStarting inference on holdout test set...")
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Testing"):
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)

            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)

            _, predicted = torch.max(outputs, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    test_loss = running_loss / len(all_labels)

    # ==========================================
    # 5. Academic Metrics & Visualization
    # ==========================================
    print("\n" + "=" * 50)
    print("🔬 FINAL CLASSICAL INFERENCE RESULTS")
    print("=" * 50)

    correct = sum(p == l for p, l in zip(all_preds, all_labels))
    accuracy = 100 * correct / len(all_labels)

    print(f"Overall Test Loss:     {test_loss:.4f}")
    print(f"Overall Test Accuracy: {accuracy:.2f}%\n")

    print("Classification Report:")
    print(
        classification_report(
            all_labels, all_preds, target_names=class_names, zero_division=0
        )
    )

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.title("Classical ResNet152 - Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")

    cm_filename = "classical_confusion_matrix.png"
    plt.savefig(cm_filename)
    print(f"\n📊 Confusion matrix plot saved as '{cm_filename}'")
    print("=" * 50)
