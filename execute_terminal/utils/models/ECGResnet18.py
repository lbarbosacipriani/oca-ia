## implementation of ResNet18 for ECG image classification with 2 classes (normal and abnormal) and timm

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

class ECGResnet18(nn.Module):
    def __init__(self, num_classes=2):
        super(ECGResnet18, self).__init__()
        # Carregar modelo ResNet18 pré-treinado
        self.model = timm.create_model('resnet18', pretrained=True)
        # Substituir a última camada totalmente conectada para classificação binária
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)