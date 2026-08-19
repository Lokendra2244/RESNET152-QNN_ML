# %% cell 1: imports
import copy
import csv
import math

import numpy as np
import pandas as pd
import pennylane as qml
import torch
from torch import nn
from torch import optim
from torchvision import models
from torchvision import transforms
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler


# %% cell 2: case check for local testing of script (same UX as your other scripts)
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

FILE_PATH = "./clean_data_set.csv" if TEST_MODE else "./clean_data_set_full.csv"

# The checkpoint saved by resnet152_baseline_fixed.py — this script picks up
# right where the classical baseline left off, rather than training a
# ResNet152 from scratch again.
CLASSICAL_CHECKPOINT = "./classical_resnet_weights_best.pth"


# %% cell 3: dataset — identical to the fixed classical script (train/eval
# transforms are still split; that bug applies here too if you skip it).
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


# %% cell 4: classical backbone — architecture-only re-declaration so we can
# load the fine-tuned checkpoint. weights=None (no ImageNet download) since
# we immediately overwrite every weight with your trained checkpoint anyway.
class ClassicalResNet(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = models.resnet152(weights=None)
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])
        self.dropout = nn.Dropout(p=0.5)
        self.classifier = nn.Linear(2048, 2)

    def forward(self, x):
        x = self.feature_extractor(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.classifier(x)
        return x


# %% cell 5: dressed quantum network — mirrors the paper's Section III.D:
#   z_pre     = tanh(W_pre * z_classical + b_pre) * (pi/2)      [2048 -> 8]
#   z_quantum = quantum_circuit(z_pre, theta)                    [8 -> 8]
#   y_hat     = W_post * z_quantum + b_post                      [8 -> 2]
#
# Gate choice follows Fig. 4 of the paper: Hadamard superposition, RY angle
# encoding of the classical features, then Q_DEPTH entangling (CNOT) +
# trainable RY layers, measured as PauliZ expectation values.
#
# Q_DEPTH isn't pinned down by the paper beyond "several layers" — 6 is a
# reasonable, commonly-used default (matches PennyLane's own quantum-transfer-
# learning reference implementation for a same-sized qubit register). Treat
# it as a hyperparameter you can sweep once the pipeline is validated.
N_QUBITS = 8
Q_DEPTH = 6

q_device = qml.device("default.qubit", wires=N_QUBITS)


def entangling_layer(n_qubits):
    # Ring-style entanglement in two half-steps (even pairs, then odd pairs)
    # so every qubit gets entangled with both neighbors each layer.
    for i in range(0, n_qubits - 1, 2):
        qml.CNOT(wires=[i, i + 1])
    for i in range(1, n_qubits - 1, 2):
        qml.CNOT(wires=[i, i + 1])


@qml.qnode(q_device, interface="torch", diff_method="backprop")
def quantum_circuit(inputs, weights):
    # inputs: (..., N_QUBITS) — note the `...` / ellipsis indexing below.
    # TorchLayer passes a *batched* tensor of shape (batch, N_QUBITS) during
    # training, so indexing with a plain inputs[i] silently indexes into the
    # batch dimension instead of the feature dimension once batch < N_QUBITS.
    # inputs[..., i] is the batch-safe way to grab feature i regardless of
    # whether a single sample or a full batch is passed through.
    for i in range(N_QUBITS):
        qml.Hadamard(wires=i)
    for i in range(N_QUBITS):
        qml.RY(inputs[..., i], wires=i)
    for layer in range(Q_DEPTH):
        entangling_layer(N_QUBITS)
        for i in range(N_QUBITS):
            qml.RY(weights[layer, i], wires=i)
    return [qml.expval(qml.PauliZ(i)) for i in range(N_QUBITS)]


class DressedQuantumNet(nn.Module):
    def __init__(self, n_qubits: int = N_QUBITS, q_depth: int = Q_DEPTH):
        super().__init__()
        self.pre_net = nn.Linear(2048, n_qubits)
        weight_shapes = {"weights": (q_depth, n_qubits)}
        self.q_layer = qml.qnn.TorchLayer(quantum_circuit, weight_shapes)
        self.post_net = nn.Linear(n_qubits, 2)

    def forward(self, x):
        x = torch.tanh(self.pre_net(x)) * (math.pi / 2)
        x = self.q_layer(x)
        x = self.post_net(x)
        return x


class HybridResNetQNN(nn.Module):
    """Frozen classical feature extractor + trainable dressed quantum head.

    This is "quantum transfer learning" in the same sense the paper uses the
    term: the classical backbone is not fine-tuned further here — it was
    already fine-tuned in the classical baseline stage. Only the small
    pre/quantum/post head (a few thousand parameters) trains now.
    """

    def __init__(self, feature_extractor: nn.Module):
        super().__init__()
        self.feature_extractor = feature_extractor
        for p in self.feature_extractor.parameters():
            p.requires_grad = False
        self.quantum_head = DressedQuantumNet()

    def forward(self, x):
        with torch.no_grad():
            features = torch.flatten(self.feature_extractor(x), 1)
        return self.quantum_head(features)

    def train(self, mode: bool = True):
        # Calling .train() on the outer model would otherwise recurse into
        # feature_extractor and flip its BatchNorm layers back to using
        # batch statistics instead of the frozen running stats — silently
        # undoing the "frozen" part even though requires_grad is False.
        super().train(mode)
        self.feature_extractor.eval()
        return self


# %% cell 6: main
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
    class_weights = 1.0 / class_counts
    sample_weights = (
        train_df["pathology"].map({i: w for i, w in enumerate(class_weights)}).values
    )
    sampler = WeightedRandomSampler(
        weights=sample_weights, num_samples=len(sample_weights), replacement=True
    )

    train_loader = DataLoader(train_data, batch_size=32, sampler=sampler)
    val_loader = DataLoader(val_data, batch_size=32, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    backbone_full = ClassicalResNet()
    backbone_full.load_state_dict(torch.load(CLASSICAL_CHECKPOINT, map_location=device))
    feature_extractor = backbone_full.feature_extractor

    model = HybridResNetQNN(feature_extractor).to(device)
    model.train()

    criterion = nn.CrossEntropyLoss()
    # Only the dressed quantum head has trainable parameters here — it's a
    # few thousand params, so a single, fairly ordinary Adam LR is fine; no
    # need for the differential-LR setup the classical backbone needed.
    optimizer = optim.Adam(model.quantum_head.parameters(), lr=2e-3)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

    # The quantum head is tiny and the backbone is frozen (forward-only), so
    # this converges in far fewer epochs than the classical baseline needed.
    num_epochs = 60
    best_val_acc = 0.0
    best_state = None

    with open("qnn_train_log.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc"])

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_true, train_preds = [], []
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

        model.eval()
        with torch.no_grad():
            val_loss = 0.0
            val_true, val_preds = [], []
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

        with open("qnn_train_log.csv", "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, avg_train_loss, train_acc, avg_val_loss, val_acc])

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())
            torch.save(best_state, "./hybrid_qnn_weights_best.pth")
            print(f"  -> new best val acc {best_val_acc:.2f}%, checkpoint saved")

    print(
        f"Best hybrid QNN validation accuracy over {num_epochs} epochs: {best_val_acc:.2f}%"
    )
    torch.save(model.state_dict(), "./hybrid_qnn_weights_final.pth")
