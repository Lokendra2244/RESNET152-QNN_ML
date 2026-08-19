# %% cell 1: imports
import copy
import csv

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
import matplotlib.pyplot as plt


# %% cell 2: Case check for local testing of script
TEST_MODE = True
while True:
    uc = input(
        "do you want to run this in test mode with 20 images?(y/n) :"
    )  # user choice
    if uc == "y":
        TEST_MODE = True
        break
    elif uc == "n":
        TEST_MODE = False
        break
    print("invalid input please try again")

if TEST_MODE:
    FILE_PATH = "./clean_data_set.csv"
else:
    FILE_PATH = "./clean_data_set_full.csv"


# %% cell 3:


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


TRAIN_TRFM = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.RandomAffine(degrees=15, scale=(0.8, 1.2), shear=17),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]
)

EVAL_TRFM = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]
)


class BreastcancerDataset(Dataset):
    def __init__(self, df: pd.DataFrame, train: bool):
        self.df = df.reset_index(drop=True)
        self.trfm = TRAIN_TRFM if train else EVAL_TRFM

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.df.iloc[idx]["cropped image file path"]
        label = self.df.iloc[idx]["pathology"]
        img = Image.open(img_path).convert("RGB")
        img = self.trfm(img)
        label = torch.tensor(label, dtype=torch.long)
        return img, label


# %% cell 4:
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
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])
        self.dropout = nn.Dropout(p=0.5)
        self.classifier = nn.Linear(2048, 2)

    def forward(self, x):
        x = self.feature_extractor(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.classifier(x)
        return x

    def trainable_backbone_params(self):
        return [p for p in self.feature_extractor.parameters() if p.requires_grad]


# %% cell 5:
def imshow(inp, title=None):
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array(IMAGENET_MEAN)
    std = np.array(IMAGENET_STD)
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.1)


# %% cell 6:
if __name__ == "__main__":
    full_df = pd.read_csv(FILE_PATH)

    train_df, val_df = train_test_split(
        full_df,
        test_size=0.2,
        stratify=full_df["pathology"],
        random_state=42,
    )
    train_data = BreastcancerDataset(train_df, train=True)
    val_data = BreastcancerDataset(val_df, train=False)

    class_counts = train_df["pathology"].value_counts().sort_index().values
    class_weights = (1.0 / class_counts).copy()
    sample_weights = (
        train_df["pathology"].map({i: w for i, w in enumerate(class_weights)}).values
    )

    samplr = WeightedRandomSampler(
        weights=sample_weights, num_samples=len(sample_weights), replacement=True
    )

    train_loader = DataLoader(train_data, batch_size=32, sampler=samplr)
    val_loader = DataLoader(val_data, batch_size=32, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ClassicalResNet(unfreeze_from="layer3").to(device)
    criterion = nn.CrossEntropyLoss()

    optimizer = optim.AdamW(
        [
            {"params": model.trainable_backbone_params(), "lr": 1e-5},
            {"params": model.classifier.parameters(), "lr": 1e-3},
        ],
        weight_decay=1e-4,
    )

    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

    # Sanity check visualization
    inputs, classes = next(iter(train_loader))
    out = torchvision.utils.make_grid(inputs)
    imshow(out, title=[x.item() for x in classes])

    # training Loop
    num_epochs = 60
    best_val_acc = 0.0
    best_state = None

    with open("train_log.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc"])

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_true = []
        train_preds = []
        for img, label in train_loader:
            img, label = img.to(device), label.to(device)
            optimizer.zero_grad()
            outputs = model(img)
            loss = criterion(outputs, label)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)

            train_true.extend(label.cpu().numpy())
            train_preds.extend(predicted.cpu().numpy())

        avg_train_loss = train_loss / len(train_loader)
        train_acc = accuracy_score(train_true, train_preds) * 100
        print(
            f"Train Epoch {epoch} | Loss: {avg_train_loss:.4f} | Accuracy: {train_acc:.2f}%"
        )

        # validation loop
        model.eval()
        with torch.no_grad():
            val_loss = 0.0
            val_true = []
            val_preds = []
            for img, label in val_loader:
                img, label = img.to(device), label.to(device)
                outputs = model(img)
                loss = criterion(outputs, label)
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)

                val_true.extend(label.cpu().numpy())
                val_preds.extend(predicted.cpu().numpy())

            avg_val_loss = val_loss / len(val_loader)
            val_acc = accuracy_score(val_true, val_preds) * 100
            print(
                f"Val Epoch {epoch} | Loss: {avg_val_loss:.4f} | Accuracy: {val_acc:.2f}%"
            )

            report = classification_report(
                val_true,
                val_preds,
                labels=[0, 1],
                target_names=["Malignant", "Benign"],
                zero_division=0,
            )
            print(report)
            print("-" * 50)

        scheduler.step(avg_val_loss)

        with open("train_log.csv", "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, avg_train_loss, train_acc, avg_val_loss, val_acc])

        # Checkpoint the best model by val accuracy instead of only saving
        # whatever weights happen to exist after the final epoch.
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())
            torch.save(best_state, "./classical_resnet_weights_best.pth")
            print(f"  -> new best val acc {best_val_acc:.2f}%, checkpoint saved")

    print(f"Best validation accuracy over {num_epochs} epochs: {best_val_acc:.2f}%")

    torch.save(model.state_dict(), "./classical_resnet_weights.pth")
