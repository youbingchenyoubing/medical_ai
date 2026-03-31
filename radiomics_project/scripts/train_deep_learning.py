#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
深度学习端到端训练主程序

Usage:
    python train_deep_learning.py --model resnet3d --epochs 50
"""

import os
import sys
import argparse
import pandas as pd
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.deep_learning_trainer import DeepLearningTrainer
from src.utils import load_config, setup_logger

logger = setup_logger(__name__)

def main():
    parser = argparse.ArgumentParser(description='Deep Learning Training')
    parser.add_argument('--model', type=str, default='resnet3d',
                       choices=['simple3dcnn', 'resnet3d', 'densenet3d'],
                       help='Model architecture')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                       help='Config file path')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=4,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate')
    
    args = parser.parse_args()
    
    # 加载配置
    config = load_config(args.config)
    
    # 更新配置
    config['model_name'] = args.model
    config['training']['epochs'] = args.epochs
    config['training']['batch_size'] = args.batch_size
    config['learning_rate'] = args.lr
    
    logger.info("="*70)
    logger.info("Deep Learning Training Pipeline")
    logger.info("="*70)
    logger.info(f"Model: {args.model}")
    logger.info(f"Epochs: {args.epochs}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Learning rate: {args.lr}")
    
    # 准备数据
    logger.info("\nPreparing data...")
    
    # 加载临床数据
    clinical_file = config['data']['clinical_file']
    if not os.path.exists(clinical_file):
        logger.error(f"Clinical file not found: {clinical_file}")
        logger.error("Please prepare clinical data first")
        return
    
    df = pd.read_csv(clinical_file)
    
    # 分割数据
    from sklearn.model_selection import train_test_split
    
    train_df, val_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df['label']
    )
    
    # 准备路径
    processed_dir = config['data']['processed_dir']
    mask_dir = config['data']['mask_dir']
    
    train_paths = [os.path.join(processed_dir, f"{case_id}.nii.gz") 
                   for case_id in train_df['case_id']]
    train_labels = train_df['label'].tolist()
    train_masks = [os.path.join(mask_dir, f"{case_id}_mask.nii.gz") 
                   for case_id in train_df['case_id']]
    
    val_paths = [os.path.join(processed_dir, f"{case_id}.nii.gz") 
                 for case_id in val_df['case_id']]
    val_labels = val_df['label'].tolist()
    val_masks = [os.path.join(mask_dir, f"{case_id}_mask.nii.gz") 
                 for case_id in val_df['case_id']]
    
    logger.info(f"Training samples: {len(train_paths)}")
    logger.info(f"Validation samples: {len(val_paths)}")
    
    # 初始化训练器
    trainer = DeepLearningTrainer(config)
    
    # 准备数据
    trainer.prepare_data(
        train_paths, train_labels, val_paths, val_labels,
        train_masks, val_masks
    )
    
    # 训练
    trainer.train()
    
    logger.info("\n" + "="*70)
    logger.info("Training completed!")
    logger.info("="*70)

if __name__ == "__main__":
    main()
