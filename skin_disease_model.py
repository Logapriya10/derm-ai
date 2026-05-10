import torch
import torch.nn as nn
from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights

CLASSES     = ["melanoma", "vitiligo", "psoriasis", "ringworm", "acne", "normal"]
NUM_CLASSES = 6
SYMPTOM_DIM = 12


class ImageBranch(nn.Module):
    def __init__(self):
        super().__init__()
        base = efficientnet_b3(weights=EfficientNet_B3_Weights.IMAGENET1K_V1)
        in_f = base.classifier[1].in_features
        base.classifier = nn.Sequential(
            nn.Dropout(0.4), nn.Linear(in_f, 512),
            nn.SiLU(), nn.Dropout(0.3), nn.Linear(512, NUM_CLASSES),
        )
        self.features   = base.features
        self.avgpool    = base.avgpool
        self.classifier = base.classifier

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


class SymptomMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(12, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(32, NUM_CLASSES),
        )

    def forward(self, x):
        return self.fc(x)


class FusionHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(NUM_CLASSES * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, NUM_CLASSES),
        )

    def forward(self, x):
        return self.fc(x)