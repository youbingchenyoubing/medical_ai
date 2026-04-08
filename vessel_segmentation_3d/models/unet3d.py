"""
3D U-Net模型实现
================

实现定制化的3D U-Net架构，用于肿瘤与血管的协同分割。

架构特点：
1. 编码器-解码器结构
2. 跳跃连接保留空间信息
3. 批归一化和ReLU激活
4. 支持多类别分割

参考论文：
- Çiçek et al. "3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation" (MICCAI 2016)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class DoubleConv3D(nn.Module):
    """
    双卷积块
    
    结构：Conv3d -> BatchNorm -> ReLU -> Conv3d -> BatchNorm -> ReLU
    
    这是U-Net的基本构建块，用于特征提取。
    
    参数:
        in_channels (int): 输入通道数
        out_channels (int): 输出通道数
        kernel_size (int): 卷积核大小，默认为3
        padding (int): 填充大小，默认为1
    
    示例:
        >>> conv_block = DoubleConv3D(64, 128)
        >>> x = torch.randn(2, 64, 32, 64, 64)
        >>> output = conv_block(x)
        >>> print(output.shape)  # torch.Size([2, 128, 32, 64, 64])
    """
    
    def __init__(self, in_channels: int, out_channels: int, 
                 kernel_size: int = 3, padding: int = 1):
        super(DoubleConv3D, self).__init__()
        
        self.double_conv = nn.Sequential(
            # 第一个卷积层
            nn.Conv3d(in_channels, out_channels, 
                     kernel_size=kernel_size, padding=padding, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            
            # 第二个卷积层
            nn.Conv3d(out_channels, out_channels, 
                     kernel_size=kernel_size, padding=padding, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        参数:
            x: 输入张量 (B, C_in, D, H, W)
        
        返回:
            输出张量 (B, C_out, D, H, W)
        """
        return self.double_conv(x)


class Down3D(nn.Module):
    """
    下采样模块
    
    结构：MaxPool3d -> DoubleConv3D
    
    用于编码器路径，降低空间分辨率，增加特征通道数。
    
    参数:
        in_channels (int): 输入通道数
        out_channels (int): 输出通道数
    
    示例:
        >>> down = Down3D(64, 128)
        >>> x = torch.randn(2, 64, 32, 64, 64)
        >>> output = down(x)
        >>> print(output.shape)  # torch.Size([2, 128, 16, 32, 32])
    """
    
    def __init__(self, in_channels: int, out_channels: int):
        super(Down3D, self).__init__()
        
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool3d(kernel_size=2, stride=2),
            DoubleConv3D(in_channels, out_channels)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        参数:
            x: 输入张量 (B, C_in, D, H, W)
        
        返回:
            输出张量 (B, C_out, D//2, H//2, W//2)
        """
        return self.maxpool_conv(x)


class Up3D(nn.Module):
    """
    上采样模块
    
    结构：ConvTranspose3d -> Concat -> DoubleConv3D
    
    用于解码器路径，增加空间分辨率，融合跳跃连接特征。
    
    参数:
        in_channels (int): 输入通道数（来自更深层）
        out_channels (int): 输出通道数
        bilinear (bool): 是否使用双线性插值，默认False使用转置卷积
    
    示例:
        >>> up = Up3D(128, 64)
        >>> x1 = torch.randn(2, 128, 16, 32, 32)  # 来自更深层
        >>> x2 = torch.randn(2, 64, 32, 64, 64)   # 跳跃连接
        >>> output = up(x1, x2)
        >>> print(output.shape)  # torch.Size([2, 64, 32, 64, 64])
    """
    
    def __init__(self, in_channels: int, out_channels: int, bilinear: bool = False):
        super(Up3D, self).__init__()
        
        if bilinear:
            # 使用双线性插值
            self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
            self.conv = DoubleConv3D(in_channels, out_channels)
        else:
            # 使用转置卷积
            self.up = nn.ConvTranspose3d(in_channels, in_channels // 2, 
                                         kernel_size=2, stride=2)
            self.conv = DoubleConv3D(in_channels, out_channels)
    
    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        参数:
            x1: 来自更深层的特征 (B, C1, D, H, W)
            x2: 跳跃连接特征 (B, C2, 2*D, 2*H, 2*W)
        
        返回:
            输出张量 (B, C_out, 2*D, 2*H, 2*W)
        """
        # 上采样
        x1 = self.up(x1)
        
        # 处理尺寸不匹配（padding）
        diff_d = x2.size()[2] - x1.size()[2]
        diff_h = x2.size()[3] - x1.size()[3]
        diff_w = x2.size()[4] - x1.size()[4]
        
        x1 = F.pad(x1, [diff_w // 2, diff_w - diff_w // 2,
                        diff_h // 2, diff_h - diff_h // 2,
                        diff_d // 2, diff_d - diff_d // 2])
        
        # 拼接跳跃连接
        x = torch.cat([x2, x1], dim=1)
        
        # 卷积
        return self.conv(x)


class UNet3D(nn.Module):
    """
    3D U-Net模型
    
    用于医学图像分割的编码器-解码器网络。
    
    架构:
        编码器: 4层下采样 (64 -> 128 -> 256 -> 512 -> 1024)
        解码器: 4层上采样 (1024 -> 512 -> 256 -> 128 -> 64)
        输出层: 1x1卷积
    
    参数:
        in_channels (int): 输入通道数，默认1（单通道CT/MRI）
        num_classes (int): 分割类别数，默认3（背景、肿瘤、血管）
        base_channels (int): 基础通道数，默认64
        dropout (float): Dropout概率，默认0.1
    
    示例:
        >>> model = UNet3D(in_channels=1, num_classes=3)
        >>> x = torch.randn(2, 1, 64, 128, 128)
        >>> output = model(x)
        >>> print(output.shape)  # torch.Size([2, 3, 64, 128, 128])
    """
    
    def __init__(self, 
                 in_channels: int = 1, 
                 num_classes: int = 3,
                 base_channels: int = 64,
                 dropout: float = 0.1):
        super(UNet3D, self).__init__()
        
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.base_channels = base_channels
        
        # 编码器路径
        self.inc = DoubleConv3D(in_channels, base_channels)
        self.down1 = Down3D(base_channels, base_channels * 2)
        self.down2 = Down3D(base_channels * 2, base_channels * 4)
        self.down3 = Down3D(base_channels * 4, base_channels * 8)
        self.down4 = Down3D(base_channels * 8, base_channels * 16)
        
        # 解码器路径
        self.up1 = Up3D(base_channels * 16, base_channels * 8)
        self.up2 = Up3D(base_channels * 8, base_channels * 4)
        self.up3 = Up3D(base_channels * 4, base_channels * 2)
        self.up4 = Up3D(base_channels * 2, base_channels)
        
        # 输出层
        self.outc = nn.Conv3d(base_channels, num_classes, kernel_size=1)
        
        # Dropout
        self.dropout = nn.Dropout3d(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        参数:
            x: 输入图像 (B, C_in, D, H, W)
        
        返回:
            logits: 分割logits (B, num_classes, D, H, W)
        """
        # 编码器路径
        x1 = self.inc(x)      # (B, 64, D, H, W)
        x2 = self.down1(x1)   # (B, 128, D/2, H/2, W/2)
        x3 = self.down2(x2)   # (B, 256, D/4, H/4, W/4)
        x4 = self.down3(x3)   # (B, 512, D/8, H/8, W/8)
        x5 = self.down4(x4)   # (B, 1024, D/16, H/16, W/16)
        
        # Dropout
        x5 = self.dropout(x5)
        
        # 解码器路径（带跳跃连接）
        x = self.up1(x5, x4)  # (B, 512, D/8, H/8, W/8)
        x = self.up2(x, x3)   # (B, 256, D/4, H/4, W/4)
        x = self.up3(x, x2)   # (B, 128, D/2, H/2, W/2)
        x = self.up4(x, x1)   # (B, 64, D, H, W)
        
        # 输出
        logits = self.outc(x)
        
        return logits
    
    def get_model_summary(self) -> str:
        """
        获取模型摘要信息
        
        返回:
            模型信息字符串
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        summary = f"""
        3D U-Net Model Summary
        =======================
        Input channels: {self.in_channels}
        Output classes: {self.num_classes}
        Base channels: {self.base_channels}
        
        Total parameters: {total_params:,}
        Trainable parameters: {trainable_params:,}
        Non-trainable parameters: {total_params - trainable_params:,}
        
        Model size: {total_params * 4 / (1024**2):.2f} MB (float32)
        """
        
        return summary


class AttentionBlock3D(nn.Module):
    """
    3D注意力块
    
    用于Attention U-Net，增强跳跃连接的特征选择能力。
    
    参数:
        F_g (int): 门控特征通道数
        F_l (int): 跳跃连接特征通道数
        F_int (int): 中间层通道数
    
    参考:
        Oktay et al. "Attention U-Net: Learning Where to Look for the Pancreas" (MIDL 2018)
    """
    
    def __init__(self, F_g: int, F_l: int, F_int: int):
        super(AttentionBlock3D, self).__init__()
        
        self.W_g = nn.Sequential(
            nn.Conv3d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(F_int)
        )
        
        self.W_x = nn.Sequential(
            nn.Conv3d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(F_int)
        )
        
        self.psi = nn.Sequential(
            nn.Conv3d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        参数:
            g: 门控特征（来自解码器）
            x: 跳跃连接特征（来自编码器）
        
        返回:
            注意力加权后的特征
        """
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        
        return x * psi


class AttentionUNet3D(nn.Module):
    """
    3D Attention U-Net
    
    在跳跃连接处添加注意力机制，提高分割精度。
    
    参数:
        in_channels (int): 输入通道数
        num_classes (int): 输出类别数
        base_channels (int): 基础通道数
    """
    
    def __init__(self, 
                 in_channels: int = 1, 
                 num_classes: int = 3,
                 base_channels: int = 64):
        super(AttentionUNet3D, self).__init__()
        
        # 编码器
        self.inc = DoubleConv3D(in_channels, base_channels)
        self.down1 = Down3D(base_channels, base_channels * 2)
        self.down2 = Down3D(base_channels * 2, base_channels * 4)
        self.down3 = Down3D(base_channels * 4, base_channels * 8)
        self.down4 = Down3D(base_channels * 8, base_channels * 16)
        
        # 注意力块
        self.att4 = AttentionBlock3D(F_g=base_channels * 16, F_l=base_channels * 8, 
                                      F_int=base_channels * 4)
        self.att3 = AttentionBlock3D(F_g=base_channels * 8, F_l=base_channels * 4, 
                                      F_int=base_channels * 2)
        self.att2 = AttentionBlock3D(F_g=base_channels * 4, F_l=base_channels * 2, 
                                      F_int=base_channels)
        self.att1 = AttentionBlock3D(F_g=base_channels * 2, F_l=base_channels, 
                                      F_int=base_channels // 2)
        
        # 解码器
        self.up1 = Up3D(base_channels * 16, base_channels * 8)
        self.up2 = Up3D(base_channels * 8, base_channels * 4)
        self.up3 = Up3D(base_channels * 4, base_channels * 2)
        self.up4 = Up3D(base_channels * 2, base_channels)
        
        # 输出
        self.outc = nn.Conv3d(base_channels, num_classes, kernel_size=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 编码器
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        # 解码器 + 注意力
        x4_att = self.att4(g=x5, x=x4)
        x = self.up1(x5, x4_att)
        
        x3_att = self.att3(g=x, x=x3)
        x = self.up2(x, x3_att)
        
        x2_att = self.att2(g=x, x=x2)
        x = self.up3(x, x2_att)
        
        x1_att = self.att1(g=x, x=x1)
        x = self.up4(x, x1_att)
        
        # 输出
        logits = self.outc(x)
        
        return logits


# 测试代码
if __name__ == "__main__":
    print("="*60)
    print("Testing 3D U-Net Models")
    print("="*60)
    
    # 测试UNet3D
    model = UNet3D(in_channels=1, num_classes=3, base_channels=32)
    print(model.get_model_summary())
    
    # 测试前向传播
    batch_size = 2
    in_channels = 1
    depth, height, width = 64, 128, 128
    
    x = torch.randn(batch_size, in_channels, depth, height, width)
    
    print(f"\nInput shape: {x.shape}")
    
    with torch.no_grad():
        output = model(x)
    
    print(f"Output shape: {output.shape}")
    
    # 测试AttentionUNet3D
    print("\n" + "="*60)
    print("Testing Attention U-Net")
    print("="*60)
    
    att_model = AttentionUNet3D(in_channels=1, num_classes=3, base_channels=32)
    
    att_params = sum(p.numel() for p in att_model.parameters())
    print(f"Attention U-Net parameters: {att_params:,}")
    
    with torch.no_grad():
        att_output = att_model(x)
    
    print(f"Attention U-Net output shape: {att_output.shape}")
    
    print("\n✓ All tests passed!")
