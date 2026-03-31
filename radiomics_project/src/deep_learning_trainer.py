"""
深度学习训练器

支持：
- 端到端训练
- 迁移学习
- 数据增强
- 早停
- 模型保存
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import SimpleITK as sitk
from typing import List, Tuple, Dict, Optional
import logging
from tqdm import tqdm
import matplotlib.pyplot as plt

from .deep_learning_models import get_model
from .utils import ensure_dir, setup_logger

logger = setup_logger(__name__)


class MedicalImageDataset(Dataset):
    """医学影像数据集"""
    
    def __init__(
        self,
        image_paths: List[str],
        labels: List[int],
        mask_paths: Optional[List[str]] = None,
        transform=None
    ):
        """
        初始化数据集
        
        Args:
            image_paths: 图像路径列表
            labels: 标签列表
            mask_paths: mask路径列表（可选）
            transform: 数据增强
        """
        self.image_paths = image_paths
        self.labels = labels
        self.mask_paths = mask_paths
        self.transform = transform
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        # 读取图像
        image = sitk.ReadImage(self.image_paths[idx])
        image_array = sitk.GetArrayFromImage(image)
        
        # 如果有mask，应用mask
        if self.mask_paths and self.mask_paths[idx]:
            mask = sitk.ReadImage(self.mask_paths[idx])
            mask_array = sitk.GetArrayFromImage(mask)
            image_array = image_array * (mask_array > 0)
        
        # 归一化
        image_array = (image_array - image_array.min()) / (image_array.max() - image_array.min())
        
        # 转换为tensor (D, H, W) -> (1, D, H, W)
        image_tensor = torch.from_numpy(image_array).float().unsqueeze(0)
        
        # 数据增强
        if self.transform:
            image_tensor = self.transform(image_tensor)
        
        label = self.labels[idx]
        
        return image_tensor, label


class DeepLearningTrainer:
    """深度学习训练器"""
    
    def __init__(self, config: dict):
        """
        初始化训练器
        
        Args:
            config: 配置字典
        """
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        logger.info(f"Using device: {self.device}")
        
        # 模型参数
        self.model_name = config.get('model_name', 'resnet3d')
        self.in_channels = config.get('in_channels', 1)
        self.num_classes = config.get('num_classes', 2)
        
        # 训练参数
        self.batch_size = config['training']['batch_size']
        self.epochs = config['training']['epochs']
        self.learning_rate = config.get('learning_rate', 1e-4)
        self.early_stopping_patience = config['training']['early_stopping_patience']
        
        # 初始化模型
        self.model = get_model(
            self.model_name,
            self.in_channels,
            self.num_classes
        ).to(self.device)
        
        # 损失函数和优化器
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5
        )
        
        # 记录
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []
        self.best_val_loss = float('inf')
        self.patience_counter = 0
    
    def prepare_data(
        self,
        train_paths: List[str],
        train_labels: List[int],
        val_paths: List[str],
        val_labels: List[int],
        train_masks: Optional[List[str]] = None,
        val_masks: Optional[List[str]] = None
    ) -> None:
        """
        准备数据
        
        Args:
            train_paths: 训练集图像路径
            train_labels: 训练集标签
            val_paths: 验证集图像路径
            val_labels: 验证集标签
            train_masks: 训练集mask路径
            val_masks: 验证集mask路径
        """
        logger.info("Preparing data...")
        
        # 创建数据集
        train_dataset = MedicalImageDataset(train_paths, train_labels, train_masks)
        val_dataset = MedicalImageDataset(val_paths, val_labels, val_masks)
        
        # 创建数据加载器
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        logger.info(f"Training samples: {len(train_dataset)}")
        logger.info(f"Validation samples: {len(val_dataset)}")
    
    def train_epoch(self) -> Tuple[float, float]:
        """训练一个epoch"""
        self.model.train()
        
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in tqdm(self.train_loader, desc="Training"):
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # 前向传播
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            # 反向传播
            loss.backward()
            self.optimizer.step()
            
            # 统计
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = correct / total
        
        return epoch_loss, epoch_acc
    
    def validate(self) -> Tuple[float, float]:
        """验证"""
        self.model.eval()
        
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in tqdm(self.val_loader, desc="Validation"):
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        epoch_loss = running_loss / len(self.val_loader)
        epoch_acc = correct / total
        
        return epoch_loss, epoch_acc
    
    def train(self) -> None:
        """完整训练流程"""
        logger.info("Starting training...")
        logger.info(f"Model: {self.model_name}")
        logger.info(f"Epochs: {self.epochs}")
        logger.info(f"Batch size: {self.batch_size}")
        
        for epoch in range(self.epochs):
            logger.info(f"\nEpoch {epoch+1}/{self.epochs}")
            logger.info("-" * 50)
            
            # 训练
            train_loss, train_acc = self.train_epoch()
            self.train_losses.append(train_loss)
            self.train_accs.append(train_acc)
            
            # 验证
            val_loss, val_acc = self.validate()
            self.val_losses.append(val_loss)
            self.val_accs.append(val_acc)
            
            # 学习率调整
            self.scheduler.step(val_loss)
            
            # 打印结果
            logger.info(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            logger.info(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            
            # 早停检查
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                self.save_model("best_model.pth")
                logger.info("Best model saved!")
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.early_stopping_patience:
                    logger.info(f"Early stopping triggered after {epoch+1} epochs")
                    break
        
        logger.info("Training completed!")
        
        # 绘制训练曲线
        self.plot_training_curves()
    
    def save_model(self, filename: str) -> None:
        """
        保存模型
        
        Args:
            filename: 文件名
        """
        ensure_dir("results/models")
        path = os.path.join("results/models", filename)
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accs': self.train_accs,
            'val_accs': self.val_accs,
        }, path)
        
        logger.info(f"Model saved to {path}")
    
    def load_model(self, path: str) -> None:
        """
        加载模型
        
        Args:
            path: 模型路径
        """
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        logger.info(f"Model loaded from {path}")
    
    def plot_training_curves(self) -> None:
        """绘制训练曲线"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Loss曲线
        ax1.plot(self.train_losses, label='Train Loss')
        ax1.plot(self.val_losses, label='Val Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Accuracy曲线
        ax2.plot(self.train_accs, label='Train Acc')
        ax2.plot(self.val_accs, label='Val Acc')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Training and Validation Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        
        ensure_dir("results/figures")
        plt.savefig("results/figures/training_curves.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("Training curves saved to results/figures/training_curves.png")
    
    def predict(self, image_paths: List[str], mask_paths: Optional[List[str]] = None) -> np.ndarray:
        """
        预测
        
        Args:
            image_paths: 图像路径列表
            mask_paths: mask路径列表
            
        Returns:
            预测概率
        """
        self.model.eval()
        
        dataset = MedicalImageDataset(image_paths, [0]*len(image_paths), mask_paths)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        
        all_probs = []
        
        with torch.no_grad():
            for images, _ in loader:
                images = images.to(self.device)
                outputs = self.model(images)
                probs = torch.softmax(outputs, dim=1)
                all_probs.append(probs.cpu().numpy())
        
        return np.concatenate(all_probs, axis=0)


if __name__ == "__main__":
    # 示例使用
    config = {
        'model_name': 'simple3dcnn',
        'in_channels': 1,
        'num_classes': 2,
        'training': {
            'batch_size': 4,
            'epochs': 50,
            'early_stopping_patience': 10
        },
        'learning_rate': 1e-4
    }
    
    trainer = DeepLearningTrainer(config)
    
    # 准备数据（示例）
    # train_paths = ['data/processed/case_001.nii.gz', ...]
    # train_labels = [0, 1, ...]
    # trainer.prepare_data(train_paths, train_labels, val_paths, val_labels)
    
    # 训练
    # trainer.train()
