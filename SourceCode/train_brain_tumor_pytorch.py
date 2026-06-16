import argparse
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support


def get_args():
    parser = argparse.ArgumentParser(description="Brain Tumor MRI Classification - Replication Study")
    parser.add_argument("--data_dir", type=str, default="Dataset", help="Folder dataset berisi Training dan Testing")
    parser.add_argument("--epochs", type=int, default=8, help="Jumlah epoch")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--img_size", type=int, default=224, help="Ukuran input gambar")
    parser.add_argument("--models", nargs="+", default=["resnet50", "efficientnet_b0", "mobilenet_v2"], help="Model yang dilatih")
    return parser.parse_args()


def ensure_dirs():
    Path("Hasil_Eksperimen").mkdir(exist_ok=True)
    Path("Gambar_Grafik").mkdir(exist_ok=True)


def get_loaders(data_dir, img_size, batch_size):
    train_dir = Path(data_dir) / "Training"
    test_dir = Path(data_dir) / "Testing"
    if not train_dir.exists() or not test_dir.exists():
        raise FileNotFoundError(
            f"Folder dataset belum benar. Harus ada {train_dir} dan {test_dir}. "
            "Extract dataset Kaggle ke folder Dataset/."
        )

    train_tfms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    test_tfms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_ds = datasets.ImageFolder(train_dir, transform=train_tfms)
    test_ds = datasets.ImageFolder(test_dir, transform=test_tfms)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    return train_ds, test_ds, train_loader, test_loader


def build_model(model_name, num_classes):
    model_name = model_name.lower()
    if model_name == "resnet50":
        weights = models.ResNet50_Weights.DEFAULT
        model = models.resnet50(weights=weights)
        for p in model.parameters():
            p.requires_grad = False
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, num_classes)
        )
    elif model_name == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.DEFAULT
        model = models.efficientnet_b0(weights=weights)
        for p in model.parameters():
            p.requires_grad = False
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, num_classes)
        )
    elif model_name == "mobilenet_v2":
        weights = models.MobileNet_V2_Weights.DEFAULT
        model = models.mobilenet_v2(weights=weights)
        for p in model.parameters():
            p.requires_grad = False
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, num_classes)
        )
    else:
        raise ValueError("Model harus salah satu: resnet50, efficientnet_b0, mobilenet_v2")
    return model


def train_one_model(model_name, train_loader, test_loader, class_names, device, epochs, lr):
    print(f"\n=== Training {model_name} ===")
    model = build_model(model_name, len(class_names)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

    history = {"epoch": [], "train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_acc = 0.0
    best_path = f"Hasil_Eksperimen/best_{model_name}.pth"

    for epoch in range(1, epochs + 1):
        start = time.time()
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        for inputs, labels in tqdm(train_loader, desc=f"{model_name} Epoch {epoch}/{epochs} Train"):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)

        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for inputs, labels in tqdm(test_loader, desc=f"{model_name} Epoch {epoch}/{epochs} Valid"):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        train_loss /= train_total
        train_acc = train_correct / train_total
        val_loss /= val_total
        val_acc = val_correct / val_total

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), best_path)

        elapsed = time.time() - start
        print(f"Epoch {epoch}: train_acc={train_acc:.4f}, val_acc={val_acc:.4f}, train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, time={elapsed:.1f}s")

    pd.DataFrame(history).to_csv(f"Hasil_Eksperimen/history_{model_name}.csv", index=False)
    plot_history(history, model_name)

    model.load_state_dict(torch.load(best_path, map_location=device))
    report, metrics = evaluate_model(model, model_name, test_loader, class_names, device)
    return metrics


def evaluate_model(model, model_name, test_loader, class_names, device):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc=f"Evaluating {model_name}"):
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            y_true.extend(labels.numpy().tolist())
            y_pred.extend(preds.cpu().numpy().tolist())

    report_dict = classification_report(y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0)
    report_df = pd.DataFrame(report_dict).transpose()
    report_df.to_csv(f"Hasil_Eksperimen/classification_report_{model_name}.csv")

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.title(f"Confusion Matrix - {model_name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(f"Gambar_Grafik/confusion_matrix_{model_name}.png", dpi=300)
    plt.close()

    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
    metrics = {
        "model": model_name,
        "accuracy": acc,
        "precision_weighted": precision,
        "recall_weighted": recall,
        "f1_weighted": f1
    }
    with open(f"Hasil_Eksperimen/hasil_{model_name}.txt", "w", encoding="utf-8") as f:
        f.write(f"Model: {model_name}\n")
        f.write(f"Accuracy: {acc:.4f}\n")
        f.write(f"Precision weighted: {precision:.4f}\n")
        f.write(f"Recall weighted: {recall:.4f}\n")
        f.write(f"F1 weighted: {f1:.4f}\n\n")
        f.write(classification_report(y_true, y_pred, target_names=class_names, zero_division=0))

    print(f"\n{model_name} => Accuracy: {acc:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
    return report_df, metrics


def plot_history(history, model_name):
    epochs = history["epoch"]
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_acc"], label="Train Accuracy")
    plt.plot(epochs, history["val_acc"], label="Validation Accuracy")
    plt.title(f"Accuracy - {model_name}")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"Gambar_Grafik/grafik_accuracy_{model_name}.png", dpi=300)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_loss"], label="Train Loss")
    plt.plot(epochs, history["val_loss"], label="Validation Loss")
    plt.title(f"Loss - {model_name}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"Gambar_Grafik/grafik_loss_{model_name}.png", dpi=300)
    plt.close()


def main():
    args = get_args()
    ensure_dirs()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    train_ds, test_ds, train_loader, test_loader = get_loaders(args.data_dir, args.img_size, args.batch_size)
    class_names = train_ds.classes
    print("Class names:", class_names)
    print("Train images:", len(train_ds), "Test images:", len(test_ds))

    all_metrics = []
    for model_name in args.models:
        metrics = train_one_model(model_name, train_loader, test_loader, class_names, device, args.epochs, args.lr)
        all_metrics.append(metrics)

    comparison = pd.DataFrame(all_metrics)
    comparison.to_csv("Hasil_Eksperimen/perbandingan_model.csv", index=False)
    comparison.to_excel("Hasil_Eksperimen/perbandingan_model.xlsx", index=False)

    plt.figure(figsize=(8, 5))
    plt.bar(comparison["model"], comparison["accuracy"])
    plt.title("Perbandingan Accuracy Model")
    plt.xlabel("Model")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig("Gambar_Grafik/perbandingan_akurasi_model.png", dpi=300)
    plt.close()

    print("\nSelesai. Cek folder Hasil_Eksperimen dan Gambar_Grafik.")
    print(comparison)


if __name__ == "__main__":
    main()
