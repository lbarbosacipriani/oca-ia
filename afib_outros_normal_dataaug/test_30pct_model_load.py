#!/usr/bin/env python3
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import timm
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
IMAGE_ROOT = Path("/home/leo/Documents/ecg_classifier/dataset/database_ptbxl")
OUTPUT_DIR = ROOT / "output" / "test_results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_DIRS = [
    ROOT / "dataset_AFIB_Others" / "4000-files" / "output",
    ROOT / "output" / "modelos",
    ROOT / "output",
]


class ECGClassifierEfficientNet(nn.Module):
    def __init__(self, num_classes=1):
        super(ECGClassifierEfficientNet, self).__init__()
        # Where we define all the parts of the model
        self.base_model = timm.create_model('efficientnet_b0', pretrained=True)
        self.features = nn.Sequential(*list(self.base_model.children())[:-1])

        enet_out_size = 1280  # Saída do EfficientNet-B0
        # For binary classification com Dropout para regularização
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.7),  # Dropout para regularização (reduz overfitting)
            nn.ReLU(),
            nn.Dropout(0.5),  # Dropout adicional antes da saída
            nn.Linear(enet_out_size, 3)  # Saída para 3 classes
        )

    def forward(self, x):
        x = self.features(x)
        output = self.classifier(x)
        return output


class ECGImageDataset(Dataset):
    def __init__(self, dataframe, image_root, transform=None):
        self.dataframe = dataframe.reset_index(drop=True)
        self.image_root = image_root
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        img_path = row["path"]
        full_path = self.image_root / img_path
        image = treat_image_PIL(str(full_path), type_return=3)
        image = image.astype(np.float32) / 255.0
        image = torch.from_numpy(image)
        label = int(row["target"])
        label = torch.tensor(label, dtype=torch.long)
        return image, label


def treat_image_PIL(img_path, type_return=3):
    im = Image.open(img_path)
    width, height = im.size
    rgb = Image.Image.split(im)
    data = rgb
    b, g, r = data[0], data[1], data[2]
    newsize = (256, 256)
    b1 = b.crop((120, 500, 2100, 1600))
    g1 = g.crop((120, 500, 2100, 1600))
    r1 = r.crop((120, 500, 2100, 1600))
    im1 = b1.resize(newsize, Image.Resampling.LANCZOS).convert("L")
    im2 = g1.resize(newsize, Image.Resampling.LANCZOS).convert("L")
    im3 = r1.resize(newsize, Image.Resampling.LANCZOS).convert("L")
    if type_return == 1:
        return Image.merge("RGB", (im1, im1, im1))
    if type_return == 2:
        return np.array([im1, im1, im1], dtype=np.uint8)
    if type_return == 3:
        return np.array([im1, im2, im3], dtype=np.uint8)
    if type_return == 4:
        return np.array(im3, dtype=np.uint8)
    raise ValueError("type_return inválido")


def evaluate_model(model_path, test_loader, device):
    model = ECGClassifierEfficientNet(num_classes=3).to(device)
    state = torch.load(model_path, map_location=device)
    if isinstance(state, dict) and any(k.startswith("layer") or k.startswith("base_model") or k.startswith("classifier") for k in state.keys()):
        model.load_state_dict(state, strict=False)
    else:
        model.load_state_dict(state, strict=False)
    model.eval()

    predictions = []
    labels = []
    with torch.no_grad():
        for images, batch_labels in tqdm(test_loader, desc=f"Inferindo {model_path.name}"):
            images = images.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            predictions.append(preds)
            labels.append(batch_labels.numpy())

    preds = np.concatenate(predictions)
    targets = np.concatenate(labels)
    metrics = {
        "accuracy": float(accuracy_score(targets, preds)),
        "f1_macro": float(f1_score(targets, preds, average="macro")),
        "precision_macro": float(precision_score(targets, preds, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(targets, preds, average="macro", zero_division=0)),
        "n_test": int(len(targets)),
    }
    return metrics


def main():
    print("=" * 80)
    print("LEITURA E DIVISÃO DO DATASET")
    print("=" * 80)
   
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo usado: {device}")

    available_models = []

    for model_dir in MODEL_DIRS:
        if model_dir.exists():
            available_models.extend(sorted(model_dir.glob("*.pth")))

    available_models = sorted(dict.fromkeys(available_models))
    if not available_models:
        raise FileNotFoundError("Nenhum modelo .pth encontrado nas pastas esperadas")

    print(f"Modelos encontrados ({len(available_models)}):")
    for model_path in available_models:
        print(f" - {model_path}")

    results = {}
    fold = 0
    for model_path in available_models:
        CSV_PATH = ROOT / f"saida_123/saidas/output/test_df_{fold}.csv"
        print(f"\nCarregando dataset de teste do fold {fold} a partir de: {CSV_PATH}")
        df = pd.read_csv(CSV_PATH)
        df = df[["patient_id", "path", "AFIB", "NORMAL", "Other"]].copy()
        for col in ["AFIB", "NORMAL", "Other"]:
            df[col] = df[col].astype(bool).astype(int)
        df["target"] = np.argmax(df[["NORMAL", "Other", "AFIB"]].to_numpy(dtype=np.int8), axis=1)
        df = df.dropna(subset=["path"]).reset_index(drop=True)
        print(f"Dataset lido: {df.shape[0]} linhas e {df.shape[1]} colunas")
        print("Distribuição do target:")
        print(df["target"].value_counts().to_string())

        train_df, test_df = train_test_split(
            df,
            test_size=0.3,
            random_state=42,
            stratify=df["target"],
        )
        print(f"Treino: {len(train_df)} amostras")
        print(f"Teste: {len(test_df)} amostras")

        test_dataset = ECGImageDataset(test_df[["path", "target"]], IMAGE_ROOT)
        test_loader = DataLoader(
            test_dataset,
            batch_size=16,
            shuffle=False,
            num_workers=2,
            pin_memory=torch.cuda.is_available(),
        )


        print(f"\nCarregando modelo: {model_path}")
        metrics = evaluate_model(model_path, test_loader, device)
        results[model_path.name] = metrics
        print(json.dumps(metrics, indent=2))
        fold += 1

    results_path = OUTPUT_DIR / "test_metrics.json"
    with results_path.open("w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nResultados salvos em: {results_path}")


if __name__ == "__main__":
    main()
