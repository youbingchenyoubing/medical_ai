"""
肿瘤与血管协同分割模块
======================

实现多任务学习的协同分割网络，同时分割肿瘤和血管。

核心思想：
1. 共享编码器提取通用特征
2. 独立解码器处理不同任务
3. 损失函数加权平衡

作者：医学影像AI研究团队
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Optional


class CoSegmentationNet(nn.Module):
    """
    协同分割网络
    
    同时分割肿瘤和血管，使用多任务学习策略。
    
    架构:
        - 共享编码器：提取通用特征
        - 肿瘤解码器：专门分割肿瘤
        - 血管解码器：专门分割血管
    
    参数:
        in_channels (int): 输入通道数，默认1
        base_channels (int): 基础通道数，默认64
        dropout (float): Dropout概率，默认0.1
    
    示例:
        >>> model = CoSegmentationNet(in_channels=1)
        >>> x = torch.randn(2, 1, 64, 128, 128)
        >>> tumor_logits, vessel_logits = model(x)
        >>> print(tumor_logits.shape, vessel_logits.shape)
    """
    
    def __init__(self, 
                 in_channels: int = 1,
                 base_channels: int = 64,
                 dropout: float = 0.1):
        super(CoSegmentationNet, self).__init__()
        
        self.in_channels = in_channels
        self.base_channels = base_channels
        
        # 共享编码器
        self.encoder = nn.Sequential(
            # Block 1
            nn.Conv3d(in_channels, base_channels, 3, padding=1),
            nn.BatchNorm3d(base_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(base_channels, base_channels, 3, padding=1),
            nn.BatchNorm3d(base_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),
            
            # Block 2
            nn.Conv3d(base_channels, base_channels * 2, 3, padding=1),
            nn.BatchNorm3d(base_channels * 2),
            nn.ReLU(inplace=True),
            nn.Conv3d(base_channels * 2, base_channels * 2, 3, padding=1),
            nn.BatchNorm3d(base_channels * 2),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),
            
            # Block 3
            nn.Conv3d(base_channels * 2, base_channels * 4, 3, padding=1),
            nn.BatchNorm3d(base_channels * 4),
            nn.ReLU(inplace=True),
            nn.Conv3d(base_channels * 4, base_channels * 4, 3, padding=1),
            nn.BatchNorm3d(base_channels * 4),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),
            
            # Block 4 (bottleneck)
            nn.Conv3d(base_channels * 4, base_channels * 8, 3, padding=1),
            nn.BatchNorm3d(base_channels * 8),
            nn.ReLU(inplace=True),
            nn.Conv3d(base_channels * 8, base_channels * 8, 3, padding=1),
            nn.BatchNorm3d(base_channels * 8),
            nn.ReLU(inplace=True),
            
            # Dropout
            nn.Dropout3d(dropout)
        )
        
        # 肿瘤分割解码器
        self.tumor_decoder = nn.Sequential(
            # Up 1
            nn.ConvTranspose3d(base_channels * 8, base_channels * 4, 2, stride=2),
            nn.Conv3d(base_channels * 4, base_channels * 4, 3, padding=1),
            nn.BatchNorm3d(base_channels * 4),
            nn.ReLU(inplace=True),
            
            # Up 2
            nn.ConvTranspose3d(base_channels * 4, base_channels * 2, 2, stride=2),
            nn.Conv3d(base_channels * 2, base_channels * 2, 3, padding=1),
            nn.BatchNorm3d(base_channels * 2),
            nn.ReLU(inplace=True),
            
            # Up 3
            nn.ConvTranspose3d(base_channels * 2, base_channels, 2, stride=2),
            nn.Conv3d(base_channels, base_channels, 3, padding=1),
            nn.BatchNorm3d(base_channels),
            nn.ReLU(inplace=True),
            
            # Output
            nn.Conv3d(base_channels, 2, 1)  # 背景 + 肿瘤
        )
        
        # 血管分割解码器
        self.vessel_decoder = nn.Sequential(
            # Up 1
            nn.ConvTranspose3d(base_channels * 8, base_channels * 4, 2, stride=2),
            nn.Conv3d(base_channels * 4, base_channels * 4, 3, padding=1),
            nn.BatchNorm3d(base_channels * 4),
            nn.ReLU(inplace=True),
            
            # Up 2
            nn.ConvTranspose3d(base_channels * 4, base_channels * 2, 2, stride=2),
            nn.Conv3d(base_channels * 2, base_channels * 2, 3, padding=1),
            nn.BatchNorm3d(base_channels * 2),
            nn.ReLU(inplace=True),
            
            # Up 3
            nn.ConvTranspose3d(base_channels * 2, base_channels, 2, stride=2),
            nn.Conv3d(base_channels, base_channels, 3, padding=1),
            nn.BatchNorm3d(base_channels),
            nn.ReLU(inplace=True),
            
            # Output
            nn.Conv3d(base_channels, 2, 1)  # 背景 + 血管
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        参数:
            x: 输入图像 (B, C_in, D, H, W)
        
        返回:
            tumor_logits: 肿瘤分割logits (B, 2, D, H, W)
            vessel_logits: 血管分割logits (B, 2, D, H, W)
        """
        # 共享特征提取
        features = self.encoder(x)
        
        # 肿瘤分割
        tumor_logits = self.tumor_decoder(features)
        
        # 血管分割
        vessel_logits = self.vessel_decoder(features)
        
        return tumor_logits, vessel_logits


class CoSegmentationLoss(nn.Module):
    """
    协同分割损失函数
    
    组合肿瘤和血管的分割损失，支持不同的权重配置。
    
    参数:
        tumor_weight (float): 肿瘤损失权重
        vessel_weight (float): 血管损失权重
        dice_weight (float): Dice Loss权重
        ce_weight (float): Cross Entropy Loss权重
    
    示例:
        >>> criterion = CoSegmentationLoss(tumor_weight=1.0, vessel_weight=1.5)
        >>> loss, tumor_loss, vessel_loss = criterion(
        ...     tumor_pred, tumor_label,
        ...     vessel_pred, vessel_label
        ... )
    """
    
    def __init__(self, 
                 tumor_weight: float = 1.0,
                 vessel_weight: float = 1.0,
                 dice_weight: float = 0.5,
                 ce_weight: float = 0.5):
        super(CoSegmentationLoss, self).__init__()
        
        self.tumor_weight = tumor_weight
        self.vessel_weight = vessel_weight
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        
        # Cross Entropy Loss
        self.ce_loss = nn.CrossEntropyLoss()
    
    def dice_loss(self, 
                  preds: torch.Tensor, 
                  targets: torch.Tensor) -> torch.Tensor:
        """
        计算Dice Loss
        
        参数:
            preds: 预测logits (B, C, ...)
            targets: 目标标签 (B, ...)
        
        返回:
            dice_loss: Dice Loss值
        """
        # Softmax
        probs = torch.softmax(preds, dim=1)
        
        # One-hot编码
        num_classes = preds.shape[1]
        targets_onehot = F.one_hot(targets.long(), num_classes)
        
        # 调整维度顺序
        if len(preds.shape) == 5:  # 3D
            targets_onehot = targets_onehot.permute(0, 4, 1, 2, 3).float()
        elif len(preds.shape) == 4:  # 2D
            targets_onehot = targets_onehot.permute(0, 3, 1, 2).float()
        
        # 计算Dice
        dims = tuple(range(2, len(preds.shape)))
        intersection = torch.sum(probs * targets_onehot, dim=dims)
        cardinality = torch.sum(probs + targets_onehot, dim=dims)
        
        dice = (2.0 * intersection + 1e-6) / (cardinality + 1e-6)
        
        # 排除背景
        dice = dice[:, 1:].mean()
        
        return 1 - dice
    
    def forward(self,
                tumor_pred: torch.Tensor,
                tumor_label: torch.Tensor,
                vessel_pred: torch.Tensor,
                vessel_label: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        计算总损失
        
        参数:
            tumor_pred: 肿瘤预测 (B, 2, ...)
            tumor_label: 肿瘤标签 (B, ...)
            vessel_pred: 血管预测 (B, 2, ...)
            vessel_label: 血管标签 (B, ...)
        
        返回:
            total_loss: 总损失
            tumor_loss: 肿瘤损失
            vessel_loss: 血管损失
        """
        # 肿瘤损失
        tumor_dice = self.dice_loss(tumor_pred, tumor_label)
        tumor_ce = self.ce_loss(tumor_pred, tumor_label)
        tumor_loss = self.dice_weight * tumor_dice + self.ce_weight * tumor_ce
        
        # 血管损失
        vessel_dice = self.dice_loss(vessel_pred, vessel_label)
        vessel_ce = self.ce_loss(vessel_pred, vessel_label)
        vessel_loss = self.dice_weight * vessel_dice + self.ce_weight * vessel_ce
        
        # 总损失
        total_loss = (self.tumor_weight * tumor_loss + 
                      self.vessel_weight * vessel_loss)
        
        return total_loss, tumor_loss, vessel_loss


class TumorVesselSegmenter:
    """
    肿瘤血管分割器
    
    完整的分割流程，包括：
    1. 数据预处理
    2. 模型推理
    3. 后处理
    4. 结果保存
    
    参数:
        model_path (str): 模型权重路径
        device (str): 设备类型 ('cuda' 或 'cpu')
        tumor_threshold (float): 肿瘤分割阈值
        vessel_threshold (float): 血管分割阈值
    
    示例:
        >>> segmenter = TumorVesselSegmenter(model_path='model.pth')
        >>> tumor_mask, vessel_mask = segmenter.segment(image)
    """
    
    def __init__(self,
                 model_path: Optional[str] = None,
                 device: str = 'cuda',
                 tumor_threshold: float = 0.5,
                 vessel_threshold: float = 0.5):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.tumor_threshold = tumor_threshold
        self.vessel_threshold = vessel_threshold
        
        # 创建模型
        self.model = CoSegmentationNet(in_channels=1)
        self.model.to(self.device)
        
        # 加载权重
        if model_path:
            self.load_model(model_path)
        
        # 损失函数
        self.criterion = CoSegmentationLoss()
    
    def load_model(self, model_path: str):
        """
        加载模型权重
        
        参数:
            model_path: 模型文件路径
        """
        checkpoint = torch.load(model_path, map_location=self.device)
        
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
        
        print(f"Model loaded from {model_path}")
    
    def segment(self, 
                image: torch.Tensor,
                return_probs: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        执行分割
        
        参数:
            image: 输入图像 (B, C, D, H, W) 或 (C, D, H, W)
            return_probs: 是否返回概率图
        
        返回:
            tumor_mask: 肿瘤mask
            vessel_mask: 血管mask
            tumor_prob: 肿瘤概率图（可选）
            vessel_prob: 血管概率图（可选）
        """
        self.model.eval()
        
        # 确保输入维度正确
        if len(image.shape) == 4:
            image = image.unsqueeze(0)
        
        image = image.to(self.device)
        
        with torch.no_grad():
            # 推理
            tumor_logits, vessel_logits = self.model(image)
            
            # 获取概率
            tumor_probs = torch.softmax(tumor_logits, dim=1)[:, 1]  # 取前景概率
            vessel_probs = torch.softmax(vessel_logits, dim=1)[:, 1]
            
            # 阈值分割
            tumor_mask = (tumor_probs > self.tumor_threshold).float()
            vessel_mask = (vessel_probs > self.vessel_threshold).float()
        
        if return_probs:
            return tumor_mask, vessel_mask, tumor_probs, vessel_probs
        else:
            return tumor_mask, vessel_mask
    
    def post_process(self,
                     tumor_mask: torch.Tensor,
                     vessel_mask: torch.Tensor,
                     min_tumor_size: int = 100,
                     min_vessel_size: int = 50) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        后处理：去除小连通区域
        
        参数:
            tumor_mask: 肿瘤mask
            vessel_mask: 血管mask
            min_tumor_size: 最小肿瘤体积
            min_vessel_size: 最小血管体积
        
        返回:
            processed_tumor: 处理后的肿瘤mask
            processed_vessel: 处理后的血管mask
        """
        try:
            from scipy.ndimage import label
            
            # 转换为numpy
            tumor_np = tumor_mask.cpu().numpy()[0]
            vessel_np = vessel_mask.cpu().numpy()[0]
            
            # 去除小连通区域
            tumor_labeled, num_tumor = label(tumor_np)
            vessel_labeled, num_vessel = label(vessel_np)
            
            # 保留大区域
            for i in range(1, num_tumor + 1):
                if np.sum(tumor_labeled == i) < min_tumor_size:
                    tumor_np[tumor_labeled == i] = 0
            
            for i in range(1, num_vessel + 1):
                if np.sum(vessel_labeled == i) < min_vessel_size:
                    vessel_np[vessel_labeled == i] = 0
            
            # 转换回tensor
            processed_tumor = torch.from_numpy(tumor_np).unsqueeze(0).float()
            processed_vessel = torch.from_numpy(vessel_np).unsqueeze(0).float()
            
            return processed_tumor, processed_vessel
            
        except ImportError:
            print("scipy not available, skipping post-processing")
            return tumor_mask, vessel_mask


# 测试代码
if __name__ == "__main__":
    print("="*60)
    print("Testing Co-Segmentation Network")
    print("="*60)
    
    # 创建模型
    model = CoSegmentationNet(in_channels=1, base_channels=32)
    
    # 计算参数数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {total_params:,}")
    
    # 测试前向传播
    batch_size = 2
    x = torch.randn(batch_size, 1, 64, 128, 128)
    
    print(f"\nInput shape: {x.shape}")
    
    with torch.no_grad():
        tumor_logits, vessel_logits = model(x)
    
    print(f"Tumor logits shape: {tumor_logits.shape}")
    print(f"Vessel logits shape: {vessel_logits.shape}")
    
    # 测试损失函数
    criterion = CoSegmentationLoss(tumor_weight=1.0, vessel_weight=1.5)
    
    tumor_labels = torch.randint(0, 2, (batch_size, 64, 128, 128))
    vessel_labels = torch.randint(0, 2, (batch_size, 64, 128, 128))
    
    total_loss, tumor_loss, vessel_loss = criterion(
        tumor_logits, tumor_labels,
        vessel_logits, vessel_labels
    )
    
    print(f"\nTotal loss: {total_loss.item():.4f}")
    print(f"Tumor loss: {tumor_loss.item():.4f}")
    print(f"Vessel loss: {vessel_loss.item():.4f}")
    
    # 测试分割器
    print("\n" + "="*60)
    print("Testing TumorVesselSegmenter")
    print("="*60)
    
    segmenter = TumorVesselSegmenter(device='cpu')
    
    tumor_mask, vessel_mask = segmenter.segment(x)
    
    print(f"Tumor mask shape: {tumor_mask.shape}")
    print(f"Vessel mask shape: {vessel_mask.shape}")
    
    print("\n✓ All tests passed!")
