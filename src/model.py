import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import EfficientNet_B4_Weights

class FishClassifier(nn.Module):
    def __init__(self, num_classes=3): 
        super(FishClassifier, self).__init__()
        self.efficientnet = models.efficientnet_b4(weights=EfficientNet_B4_Weights.DEFAULT)

        self.efficientnet.classifier[1] = nn.Linear(self.efficientnet.classifier[1].in_features, num_classes)

    def forward(self, x):
        return self.efficientnet(x)