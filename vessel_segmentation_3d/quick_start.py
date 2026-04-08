"""
快速开始示例脚本
================

演示如何使用血管分割与三维重建系统。

运行方式：
    python quick_start.py

作者：医学影像AI研究团队
日期：2026-04-08
"""

import os
import sys
import numpy as np
from pathlib import Path

# 添加父目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))


def example_1_basic_usage():
    """
    示例1：基本使用流程
    """
    print("\n" + "="*70)
    print("示例1：基本使用流程")
    print("="*70)
    
    from vessel_segmentation_3d.models.unet3d import UNet3D
    import torch
    
    # 创建模型
    print("\n创建3D U-Net模型...")
    model = UNet3D(
        in_channels=1,
        num_classes=3,  # 背景、肿瘤、血管
        base_channels=32
    )
    
    # 打印模型信息
    print(model.get_model_summary())
    
    # 测试前向传播
    print("\n测试前向传播...")
    x = torch.randn(1, 1, 32, 64, 64)
    
    with torch.no_grad():
        output = model(x)
    
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")
    
    print("\n✓ 示例1完成")


def example_2_skeletonization():
    """
    示例2：血管骨架化
    """
    print("\n" + "="*70)
    print("示例2：血管骨架化")
    print("="*70)
    
    from vessel_segmentation_3d.skeletonization.morphological import (
        skeletonize_vessel_morphological,
        calculate_skeleton_quality
    )
    
    # 创建测试血管（圆柱体）
    print("\n创建测试血管...")
    vessel_mask = np.zeros((50, 50, 50), dtype=np.uint8)
    
    # 创建一个弯曲的血管
    for i in range(10, 40):
        y_center = 25 + int(5 * np.sin(i * 0.1))
        z_center = 25 + int(5 * np.cos(i * 0.1))
        
        # 创建圆形截面
        for y in range(y_center - 3, y_center + 4):
            for z in range(z_center - 3, z_center + 4):
                if (y - y_center)**2 + (z - z_center)**2 <= 9:
                    vessel_mask[i, y, z] = 1
    
    print(f"血管体素数量: {np.sum(vessel_mask)}")
    
    # 执行骨架化
    print("\n执行骨架化...")
    skeleton_points, skeleton = skeletonize_vessel_morphological(vessel_mask)
    
    print(f"骨架点数量: {len(skeleton_points)}")
    
    # 计算质量指标
    quality = calculate_skeleton_quality(skeleton, vessel_mask)
    
    print("\n骨架质量指标:")
    for key, value in quality.items():
        print(f"  {key}: {value}")
    
    print("\n✓ 示例2完成")


def example_3_topology_analysis():
    """
    示例3：拓扑结构分析
    """
    print("\n" + "="*70)
    print("示例3：拓扑结构分析")
    print("="*70)
    
    from vessel_segmentation_3d.skeletonization.topology_analysis import (
        analyze_vessel_topology,
        get_topology_statistics
    )
    
    # 创建测试骨架（Y形结构）
    print("\n创建测试骨架...")
    skeleton_points = np.array([
        [10, 10, 10],  # 起点
        [11, 10, 10],
        [12, 10, 10],
        [13, 10, 10],
        [14, 10, 10],  # 分叉点
        [15, 11, 10],  # 分支1
        [16, 12, 10],
        [17, 13, 10],  # 端点1
        [15, 9, 10],   # 分支2
        [16, 8, 10],
        [17, 7, 10],   # 端点2
    ])
    
    print(f"骨架点数量: {len(skeleton_points)}")
    
    # 分析拓扑
    print("\n分析拓扑结构...")
    graph, branches, junctions, endpoints = analyze_vessel_topology(skeleton_points)
    
    print(f"\n分叉点数量: {len(junctions)}")
    print(f"端点数量: {len(endpoints)}")
    print(f"分支数量: {len(branches)}")
    
    # 获取统计信息
    stats = get_topology_statistics(graph, branches, junctions, endpoints)
    
    print("\n拓扑统计:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print("\n✓ 示例3完成")


def example_4_curvature_torsion():
    """
    示例4：曲率和扭率计算
    """
    print("\n" + "="*70)
    print("示例4：曲率和扭率计算")
    print("="*70)
    
    from vessel_segmentation_3d.morphometry.curvature import (
        calculate_curvature_3d,
        calculate_curvature_statistics
    )
    from vessel_segmentation_3d.morphometry.torsion import (
        calculate_torsion_3d,
        calculate_torsion_statistics,
        classify_torsion_pattern
    )
    
    # 创建测试曲线1：螺旋线（右手螺旋）
    print("\n创建测试曲线1: 右手螺旋线...")
    t = np.linspace(0, 4*np.pi, 100)
    radius = 5.0
    pitch = 2.0
    points_helix = np.column_stack([
        radius * np.cos(t),
        radius * np.sin(t),
        pitch * t
    ])
    
    # 计算曲率
    curvatures = calculate_curvature_3d(points_helix, smooth=0.0)
    curvature_stats = calculate_curvature_statistics(curvatures)
    
    print("\n曲率统计:")
    for key, value in curvature_stats.items():
        print(f"  {key}: {value:.6f}")
    
    # 计算扭率
    torsions = calculate_torsion_3d(points_helix, smooth=0.0)
    torsion_stats = calculate_torsion_statistics(torsions)
    
    print("\n扭率统计:")
    for key, value in torsion_stats.items():
        print(f"  {key}: {value:.6f}")
    
    # 分类扭率模式
    pattern = classify_torsion_pattern(torsions)
    print(f"\n扭率模式: {pattern}")
    
    # 创建测试曲线2：平面曲线
    print("\n创建测试曲线2: 平面半圆...")
    theta = np.linspace(0, np.pi, 100)
    points_planar = np.column_stack([
        10 * np.cos(theta),
        10 * np.sin(theta),
        np.zeros_like(theta)
    ])
    
    torsions_planar = calculate_torsion_3d(points_planar, smooth=0.1)
    pattern_planar = classify_torsion_pattern(torsions_planar)
    
    print(f"平面曲线扭率模式: {pattern_planar}")
    
    print("\n✓ 示例4完成")


def example_5_branching_density():
    """
    示例5：分支密度计算
    """
    print("\n" + "="*70)
    print("示例5：分支密度计算")
    print("="*70)
    
    from vessel_segmentation_3d.morphometry.branching import (
        calculate_branching_density,
        calculate_vessel_density,
        calculate_branching_features
    )
    
    # 创建测试数据
    print("\n创建测试数据...")
    tumor_mask = np.zeros((50, 50, 50), dtype=np.uint8)
    tumor_mask[10:40, 10:40, 10:40] = 1
    
    vessel_mask = np.zeros((50, 50, 50), dtype=np.uint8)
    vessel_mask[15:35, 15:35, 15:35] = 1
    
    junctions = [[20, 20, 20], [25, 25, 25], [30, 30, 30]]
    branches = [
        [[20, 20, 20], [21, 21, 21], [22, 22, 22]],
        [[25, 25, 25], [26, 26, 26], [27, 27, 27]],
    ]
    
    # 计算分支密度
    print("\n计算分支密度...")
    density, volume = calculate_branching_density(
        junctions, tumor_mask, spacing=(1.0, 1.0, 1.0)
    )
    
    print(f"分支密度: {density:.6f} 分叉数/mm³")
    print(f"肿瘤体积: {volume:.2f} mm³")
    
    # 计算血管密度
    print("\n计算血管密度...")
    vessel_density, vessel_vol, tumor_vol = calculate_vessel_density(
        vessel_mask, tumor_mask, spacing=(1.0, 1.0, 1.0)
    )
    
    print(f"血管密度: {vessel_density:.4f}")
    print(f"血管体积: {vessel_vol:.2f} mm³")
    
    # 计算所有分支特征
    print("\n计算所有分支特征...")
    features = calculate_branching_features(
        junctions, branches, tumor_mask, vessel_mask, spacing=(1.0, 1.0, 1.0)
    )
    
    print("\n所有分支特征:")
    for key, value in features.items():
        print(f"  {key}: {value}")
    
    print("\n✓ 示例5完成")


def example_6_feature_extraction():
    """
    示例6：综合特征提取
    """
    print("\n" + "="*70)
    print("示例6：综合特征提取")
    print("="*70)
    
    from vessel_segmentation_3d.morphometry.feature_extractor import VesselMorphometryExtractor
    
    # 创建测试数据
    print("\n创建测试数据...")
    vessel_mask = np.zeros((50, 50, 50), dtype=np.uint8)
    vessel_mask[15:35, 15:35, 15:35] = 1
    
    tumor_mask = np.zeros((50, 50, 50), dtype=np.uint8)
    tumor_mask[10:40, 10:40, 10:40] = 1
    
    # 创建简单的骨架点
    skeleton_points = np.array([
        [20, 20, 20],
        [21, 21, 21],
        [22, 22, 22],
        [23, 23, 23],
        [24, 24, 24],
    ])
    
    branches = [
        [[20, 20, 20], [21, 21, 21], [22, 22, 22], [23, 23, 23], [24, 24, 24]]
    ]
    
    junctions = [[22, 22, 22]]
    endpoints = [[20, 20, 20], [24, 24, 24]]
    
    # 初始化提取器
    print("\n初始化特征提取器...")
    extractor = VesselMorphometryExtractor(
        spacing=(1.0, 1.0, 1.0),
        verbose=True
    )
    
    # 提取特征
    print("\n提取特征...")
    features = extractor.extract_all_features(
        vessel_mask=vessel_mask,
        tumor_mask=tumor_mask,
        skeleton_points=skeleton_points,
        branches=branches,
        junctions=junctions,
        endpoints=endpoints
    )
    
    # 打印特征
    print("\n提取的特征:")
    print("="*60)
    
    for key, value in sorted(features.items()):
        if isinstance(value, float):
            print(f"{key:30s}: {value:.6f}")
        else:
            print(f"{key:30s}: {value}")
    
    # 保存特征
    output_path = 'test_features.csv'
    extractor.save_features(features, output_path, case_id='test_case')
    
    print(f"\n特征已保存到: {output_path}")
    
    print("\n✓ 示例6完成")


def main():
    """主函数"""
    print("\n" + "="*70)
    print("血管分割与三维重建系统 - 快速开始示例")
    print("="*70)
    
    print("\n本脚本将演示以下功能:")
    print("1. 基本使用流程（3D U-Net模型）")
    print("2. 血管骨架化")
    print("3. 拓扑结构分析")
    print("4. 曲率和扭率计算")
    print("5. 分支密度计算")
    print("6. 综合特征提取")
    
    input("\n按Enter键开始...")
    
    # 运行所有示例
    try:
        example_1_basic_usage()
        input("\n按Enter键继续下一个示例...")
        
        example_2_skeletonization()
        input("\n按Enter键继续下一个示例...")
        
        example_3_topology_analysis()
        input("\n按Enter键继续下一个示例...")
        
        example_4_curvature_torsion()
        input("\n按Enter键继续下一个示例...")
        
        example_5_branching_density()
        input("\n按Enter键继续下一个示例...")
        
        example_6_feature_extraction()
        
    except Exception as e:
        print(f"\n错误: {str(e)}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*70)
    print("所有示例完成!")
    print("="*70)
    
    print("\n下一步:")
    print("1. 查看README.md了解详细使用方法")
    print("2. 准备自己的医学影像数据")
    print("3. 使用pipeline.py运行完整流程")
    
    print("\n感谢使用血管分割与三维重建系统!")


if __name__ == "__main__":
    main()
