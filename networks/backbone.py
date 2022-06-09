import torch.nn as nn
from torchvision import models


resnet_dict = {"ResNet18": models.resnet18, "ResNet34": models.resnet34, "ResNet50": models.resnet50,
               "ResNet101": models.resnet101, "ResNet152": models.resnet152}


class ResNet(nn.Module):
    def __init__(self, name, num_classes) -> None:
        super().__init__()
        self.model = resnet_dict[name](num_classes=num_classes)
    
    def forward(self, x):
        return self.model(x)