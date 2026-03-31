"""
深度学习端到端模型

支持多种架构：
- 3D CNN
- ResNet-3D
- DenseNet-3D
- MedicalNet（预训练）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import numpy as np

class Simple3DCNN(nn.Module):
    """简单的3D CNN网络"""
    
    def __init__(self, in_channels: int = 1, num_classes: int = 2, dropout: float = 0.5):
        super(Simple3DCNN, self).__init__()
        
        self.features = nn.Sequential(
            # Block 1
            nn.Conv3d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2),
            
            # Block 2
            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2),
            
            # Block 3
            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2),
            
            # Block 4
            nn.Conv3d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d(1),
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class ResBlock3D(nn.Module):
    """3D残差块"""
    
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super(ResBlock3D, self).__init__()
        
        self.conv1 = nn.Conv3d(
            in_channels, out_channels, 
            kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv3d(
            out_channels, out_channels, 
            kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm3d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(out_channels)
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += identity
        out = self.relu(out)
        
        return out


class ResNet3D(nn.Module):
    """3D ResNet网络"""
    
    def __init__(
        self, 
        in_channels: int = 1, 
        num_classes: int = 2,
        layers: Tuple[int, int, int, int] = (2, 2, 2, 2)
    ):
        super(ResNet3D, self).__init__()
        
        self.in_channels = 64
        
        self.conv1 = nn.Conv3d(
            in_channels, 64, 
            kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(64, layers[0], stride=1)
        self.layer2 = self._make_layer(128, layers[1], stride=2)
        self.layer3 = self._make_layer(256, layers[2], stride=2)
        self.layer4 = self._make_layer(512, layers[3], stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Linear(512, num_classes)
    
    def _make_layer(self, out_channels: int, blocks: int, stride: int) -> nn.Sequential:
        layers = []
        layers.append(ResBlock3D(self.in_channels, out_channels, stride))
        self.in_channels = out_channels
        
        for _ in range(1, blocks):
            layers.append(ResBlock3D(out_channels, out_channels))
        
        return nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        
        return x


class DenseBlock3D(nn.Module):
    """3D DenseBlock"""
    
    def __init__(self, in_channels: int, growth_rate: int, num_layers: int):
        super(DenseBlock3D, self).__init__()
        
        layers = []
        for i in range(num_layers):
            layers.append(self._make_layer(in_channels + i * growth_rate, growth_rate))
        
        self.block = nn.Sequential(*layers)
    
    def _make_layer(self, in_channels: int, growth_rate: int) -> nn.Sequential:
        return nn.Sequential(
            nn.BatchNorm3d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels, growth_rate, kernel_size=3, padding=1, bias=False)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.block:
            new_features = layer(x)
            x = torch.cat([x, new_features], dim=1)
        return x


class DenseNet3D(nn.Module):
    """3D DenseNet网络"""
    
    def __init__(
        self, 
        in_channels: int = 1, 
        num_classes: int = 2,
        growth_rate: int = 32,
        block_config: Tuple[int, int, int, int] = (6, 12, 24, 16)
    ):
        super(DenseNet3D, self).__init__()
        
        # Initial convolution
        self.features = nn.Sequential(
            nn.Conv3d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        )
        
        # Dense blocks
        num_features = 64
        for i, num_layers in enumerate(block_config):
            block = DenseBlock3D(num_features, growth_rate, num_layers)
            self.features.add_module(f'denseblock{i+1}', block)
            num_features = num_features + num_layers * growth_rate
            
            if i != len(block_config) - 1:
                trans = nn.Sequential(
                    nn.BatchNorm3d(num_features),
                    nn.ReLU(inplace=True),
                    nn.Conv3d(num_features, num_features // 2, kernel_size=1, bias=False),
                    nn.AvgPool3d(kernel_size=2, stride=2)
                )
                self.features.add_module(f'transition{i+1}', trans)
                num_features = num_features // 2
        
        # Final batch norm
        self.features.add_module('norm5', nn.BatchNorm3d(num_features))
        
        # Classifier
        self.classifier = nn.Linear(num_features, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool3d(out, 1)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out


def get_model(
    model_name: str = 'resnet3d',
    in_channels: int = 1,
    num_classes: int = 2,
    pretrained: bool = False
) -> nn.Module:
    """
    获取模型
    
    Args:
        model_name: 模型名称 ('simple3dcnn', 'resnet3d', 'densenet3d')
        in_channels: 输入通道数
        num_classes: 分类数量
        pretrained: 是否使用预训练权重
        
    Returns:
        模型
    """
    model_dict = {
        'simple3dcnn': Simple3DCNN,
        'resnet3d': ResNet3D,
        'densenet3d': DenseNet3D
    }
    
    if model_name not in model_dict:
        raise ValueError(f"Unknown model: {model_name}")
    
    model = model_dict[model_name](in_channels, num_classes)
    
    if pretrained:
        # TODO: 加载预训练权重
        pass
    
    return model


if __name__ == "__main__":
    # 测试模型
    batch_size = 2
    in_channels = 1
    depth, height, width = 64, 128, 128
    
    x = torch.randn(batch_size, in_channels, depth, height, width)
    
    # 测试不同模型
    models = ['simple3dcnn', 'resnet3d', 'densenet3d']
    
    for model_name in models:
        print(f"\nTesting {model_name}...")
        model = get_model(model_name, in_channels, num_classes=2)
        
        # 计算参数数量
        num_params = sum(p.numel() for p in model.parameters())
        print(f"Number of parameters: {num_params:,}")
        
        # 前向传播
        with torch.no_grad():
            output = model(x)
        
        print(f"Input shape: {x.shape}")
        print(f"Output shape: {output.shape}")
