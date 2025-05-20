import torch.nn as nn
import torch
from torchvision import models

from project.model.attention.cbam import CBAM


#  Định nghĩa CBAM ResNet
class CBAMResNet(nn.Module):
    def __init__(self, num_classes=2, pretrained=True):
        super(CBAMResNet, self).__init__()
        # Tải ResNet50 đã được pretrained
        self.resnet = models.resnet50(pretrained=pretrained)

        # Thêm CBAM vào các layer của ResNet
        self.cbam1 = CBAM(256)
        self.cbam2 = CBAM(512)
        self.cbam3 = CBAM(1024)
        self.cbam4 = CBAM(2048)

        # Sửa đổi forward của ResNet
        in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(in_features, 512)  # Feature extraction layer
        self.relu = nn.ReLU()
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x):
        # ResNet50 layers
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        # Layer 1 với CBAM
        x = self.resnet.layer1(x)
        x = self.cbam1(x)

        # Layer 2 với CBAM
        x = self.resnet.layer2(x)
        x = self.cbam2(x)

        # Layer 3 với CBAM
        x = self.resnet.layer3(x)
        x = self.cbam3(x)

        # Layer 4 với CBAM
        x = self.resnet.layer4(x)
        x = self.cbam4(x)

        # Feature extraction
        x = self.resnet.avgpool(x)
        x = torch.flatten(x, 1)
        features = self.resnet.fc(x)

        # Classification
        output = self.classifier(features)

        return features, output