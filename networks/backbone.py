import torch.nn as nn
from torchvision import models


resnet_dict = {"ResNet18": models.resnet18, "ResNet34": models.resnet34, "ResNet50": models.resnet50,
               "ResNet101": models.resnet101, "ResNet152": models.resnet152}


class SimpleCNN(nn.Module):
    def __init__(self, input_channels, output_num):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=input_channels, out_channels=16, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )
        fc1_input_features = 512

        self.fc1 = nn.Sequential(
            nn.Linear(in_features=fc1_input_features, out_features=512),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(in_features=512, out_features=output_num),
            nn.Softmax(dim=-1)
        )
        self.dropout = nn.Dropout(p=.5)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)

        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x


class ResNet(nn.Module):
    def __init__(self, name, num_classes) -> None:
        super().__init__()
        self.model = resnet_dict[name](num_classes=num_classes)
    
    def forward(self, x):
        return self.model(x)