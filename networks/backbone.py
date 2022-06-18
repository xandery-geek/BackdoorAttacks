import torch
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


class ResNetFeature(nn.Module):
    def __init__(self, name, pretrained=False) -> None:
        super().__init__()
        model_resnet = resnet_dict[name](pretrained=pretrained)

        self.conv1 = model_resnet.conv1
        self.bn1 = model_resnet.bn1
        self.relu = model_resnet.relu
        self.maxpool = model_resnet.maxpool
        self.layer1 = model_resnet.layer1
        self.layer2 = model_resnet.layer2
        self.layer3 = model_resnet.layer3
        self.layer4 = model_resnet.layer4
        self.feature_layers = nn.Sequential(self.conv1, self.bn1, self.relu, self.maxpool, self.layer1, self.layer2,
                                            self.layer3, self.layer4)
        
        self.mean = torch.tensor([[[0.485]], [[0.456]], [[0.406]]]).cuda()
        self.std = torch.tensor([[[0.229]], [[0.224]], [[0.225]]]).cuda()
    
    def copy_from_resnet(self, state_dict):
        new_state_dict = {}
        for k, v in state_dict.items():
            k = k.replace('model.', '')
            if not k.startswith('fc.'):
                new_state_dict[k] = v

        self.load_state_dict(new_state_dict, strict=False)

    def forward(self, x):
        x = (x - self.mean) / self.std
        f = self.feature_layers(x)
        f = f.view(f.size(0), -1)

        return f