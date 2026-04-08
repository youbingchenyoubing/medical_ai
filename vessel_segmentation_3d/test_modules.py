"""
模块测试脚本
============

测试所有核心模块是否正常工作。

运行方式：
    python test_modules.py

作者：医学影像AI研究团队
日期：2026-04-08
"""

import sys
from pathlib import Path
import numpy as np

# 添加父目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_imports():
    """测试模块导入"""
    print("\n" + "="*60)
    print("测试1: 模块导入")
    print("="*60)
    
    try:
        print("\n导入模型模块...")
        from vessel_segmentation_3d.models.unet3d import UNet3D
        from vessel_segmentation_3d.models.nnunet import nnUNetSegmenter
        print("✓ 模型模块导入成功")
        
        print("\n导入分割模块...")
        from vessel_segmentation_3d.segmentation.tumor_vessel_seg import (
            CoSegmentationNet, TumorVesselSegmenter
        )
        print("✓ 分割模块导入成功")
        
        print("\n导入骨架化模块...")
        from vessel_segmentation_3d.skeletonization.morphological import (
            skeletonize_vessel_morphological
        )
        from vessel_segmentation_3d.skeletonization.topology_analysis import (
            analyze_vessel_topology
        )
        print("✓ 骨架化模块导入成功")
        
        print("\n导入形态量化模块...")
        from vessel_segmentation_3d.morphometry.curvature import calculate_curvature_3d
        from vessel_segmentation_3d.morphometry.torsion import calculate_torsion_3d
        from vessel_segmentation_3d.morphometry.branching import calculate_branching_density
        from vessel_segmentation_3d.morphometry.feature_extractor import VesselMorphometryExtractor
        print("✓ 形态量化模块导入成功")
        
        print("\n导入主流程...")
        from vessel_segmentation_3d.pipeline import VesselSegmentationReconstructionPipeline
        print("✓ 主流程导入成功")
        
        return True
        
    except Exception as e:
        print(f"\n✗ 导入失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_models():
    """测试模型"""
    print("\n" + "="*60)
    print("测试2: 模型测试")
    print("="*60)
    
    try:
        import torch
        from vessel_segmentation_3d.models.unet3d import UNet3D
        
        print("\n创建3D U-Net模型...")
        model = UNet3D(in_channels=1, num_classes=3, base_channels=16)
        
        print(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
        
        print("\n测试前向传播...")
        x = torch.randn(1, 1, 32, 64, 64)
        
        with torch.no_grad():
            output = model(x)
        
        print(f"输入形状: {x.shape}")
        print(f"输出形状: {output.shape}")
        
        assert output.shape == (1, 3, 32, 64, 64), "输出形状不正确"
        
        print("\n✓ 模型测试通过")
        return True
        
    except Exception as e:
        print(f"\n✗ 模型测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_skeletonization():
    """测试骨架化"""
    print("\n" + "="*60)
    print("测试3: 骨架化测试")
    print("="*60)
    
    try:
        from vessel_segmentation_3d.skeletonization.morphological import (
            skeletonize_vessel_morphological
        )
        
        print("\n创建测试血管...")
        vessel_mask = np.zeros((30, 30, 30), dtype=np.uint8)
        
        # 创建简单的血管
        for i in range(10, 20):
            for j in range(12, 18):
                for k in range(12, 18):
                    vessel_mask[i, j, k] = 1
        
        print(f"血管体素数量: {np.sum(vessel_mask)}")
        
        print("\n执行骨架化...")
        skeleton_points, skeleton = skeletonize_vessel_morphological(vessel_mask)
        
        print(f"骨架点数量: {len(skeleton_points)}")
        
        assert len(skeleton_points) > 0, "骨架点数量为0"
        
        print("\n✓ 骨架化测试通过")
        return True
        
    except Exception as e:
        print(f"\n✗ 骨架化测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_topology():
    """测试拓扑分析"""
    print("\n" + "="*60)
    print("测试4: 拓扑分析测试")
    print("="*60)
    
    try:
        from vessel_segmentation_3d.skeletonization.topology_analysis import (
            analyze_vessel_topology
        )
        
        print("\n创建测试骨架...")
        skeleton_points = np.array([
            [10, 10, 10],
            [11, 10, 10],
            [12, 10, 10],
            [13, 10, 10],
            [14, 10, 10],
        ])
        
        print(f"骨架点数量: {len(skeleton_points)}")
        
        print("\n分析拓扑...")
        graph, branches, junctions, endpoints = analyze_vessel_topology(skeleton_points)
        
        print(f"分支数量: {len(branches)}")
        print(f"分叉点数量: {len(junctions)}")
        print(f"端点数量: {len(endpoints)}")
        
        assert graph.number_of_nodes() == len(skeleton_points), "图节点数量不正确"
        
        print("\n✓ 拓扑分析测试通过")
        return True
        
    except Exception as e:
        print(f"\n✗ 拓扑分析测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_curvature():
    """测试曲率计算"""
    print("\n" + "="*60)
    print("测试5: 曲率计算测试")
    print("="*60)
    
    try:
        from vessel_segmentation_3d.morphometry.curvature import calculate_curvature_3d
        
        print("\n创建测试曲线...")
        # 创建半圆
        theta = np.linspace(0, np.pi, 50)
        radius = 10.0
        points = np.column_stack([
            radius * np.cos(theta),
            radius * np.sin(theta),
            np.zeros_like(theta)
        ])
        
        print(f"理论曲率: {1/radius:.4f}")
        
        print("\n计算曲率...")
        curvatures = calculate_curvature_3d(points, smooth=0.0)
        
        print(f"计算的平均曲率: {np.mean(curvatures):.4f}")
        print(f"曲率标准差: {np.std(curvatures):.4f}")
        
        assert len(curvatures) == len(points), "曲率数量不正确"
        
        print("\n✓ 曲率计算测试通过")
        return True
        
    except Exception as e:
        print(f"\n✗ 曲率计算测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_torsion():
    """测试扭率计算"""
    print("\n" + "="*60)
    print("测试6: 扭率计算测试")
    print("="*60)
    
    try:
        from vessel_segmentation_3d.morphometry.torsion import (
            calculate_torsion_3d,
            classify_torsion_pattern
        )
        
        print("\n创建测试曲线（螺旋线）...")
        t = np.linspace(0, 4*np.pi, 100)
        points = np.column_stack([
            5 * np.cos(t),
            5 * np.sin(t),
            2 * t
        ])
        
        print("\n计算扭率...")
        torsions = calculate_torsion_3d(points, smooth=0.0)
        
        print(f"平均扭率: {np.mean(torsions):.4f}")
        print(f"扭率模式: {classify_torsion_pattern(torsions)}")
        
        assert len(torsions) == len(points), "扭率数量不正确"
        
        print("\n✓ 扭率计算测试通过")
        return True
        
    except Exception as e:
        print(f"\n✗ 扭率计算测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_branching():
    """测试分支密度计算"""
    print("\n" + "="*60)
    print("测试7: 分支密度测试")
    print("="*60)
    
    try:
        from vessel_segmentation_3d.morphometry.branching import calculate_branching_density
        
        print("\n创建测试数据...")
        tumor_mask = np.zeros((50, 50, 50), dtype=np.uint8)
        tumor_mask[10:40, 10:40, 10:40] = 1
        
        junctions = [[20, 20, 20], [25, 25, 25]]
        
        print("\n计算分支密度...")
        density, volume = calculate_branching_density(
            junctions, tumor_mask, spacing=(1.0, 1.0, 1.0)
        )
        
        print(f"分支密度: {density:.6f} 分叉数/mm³")
        print(f"肿瘤体积: {volume:.2f} mm³")
        
        assert density > 0, "分支密度应该大于0"
        
        print("\n✓ 分支密度测试通过")
        return True
        
    except Exception as e:
        print(f"\n✗ 分支密度测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_feature_extractor():
    """测试特征提取器"""
    print("\n" + "="*60)
    print("测试8: 特征提取器测试")
    print("="*60)
    
    try:
        from vessel_segmentation_3d.morphometry.feature_extractor import VesselMorphometryExtractor
        
        print("\n创建测试数据...")
        vessel_mask = np.zeros((50, 50, 50), dtype=np.uint8)
        vessel_mask[15:35, 15:35, 15:35] = 1
        
        tumor_mask = np.zeros((50, 50, 50), dtype=np.uint8)
        tumor_mask[10:40, 10:40, 10:40] = 1
        
        skeleton_points = np.array([
            [20, 20, 20],
            [21, 21, 21],
            [22, 22, 22],
            [23, 23, 23],
            [24, 24, 24],
        ])
        
        branches = [[[20, 20, 20], [21, 21, 21], [22, 22, 22], [23, 23, 23], [24, 24, 24]]]
        junctions = [[22, 22, 22]]
        endpoints = [[20, 20, 20], [24, 24, 24]]
        
        print("\n初始化特征提取器...")
        extractor = VesselMorphometryExtractor(spacing=(1.0, 1.0, 1.0), verbose=False)
        
        print("\n提取特征...")
        features = extractor.extract_all_features(
            vessel_mask=vessel_mask,
            tumor_mask=tumor_mask,
            skeleton_points=skeleton_points,
            branches=branches,
            junctions=junctions,
            endpoints=endpoints
        )
        
        print(f"\n提取特征数量: {len(features)}")
        
        assert len(features) > 20, "特征数量不足"
        
        print("\n部分特征:")
        for i, (key, value) in enumerate(sorted(features.items())):
            if i >= 10:
                break
            if isinstance(value, float):
                print(f"  {key}: {value:.6f}")
        
        print("\n✓ 特征提取器测试通过")
        return True
        
    except Exception as e:
        print(f"\n✗ 特征提取器测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主函数"""
    print("\n" + "="*70)
    print("血管分割与三维重建系统 - 模块测试")
    print("="*70)
    
    print("\n将测试以下模块:")
    print("1. 模块导入")
    print("2. 模型测试")
    print("3. 骨架化测试")
    print("4. 拓扑分析测试")
    print("5. 曲率计算测试")
    print("6. 扭率计算测试")
    print("7. 分支密度测试")
    print("8. 特征提取器测试")
    
    # 运行测试
    results = []
    
    results.append(("模块导入", test_imports()))
    results.append(("模型测试", test_models()))
    results.append(("骨架化测试", test_skeletonization()))
    results.append(("拓扑分析测试", test_topology()))
    results.append(("曲率计算测试", test_curvature()))
    results.append(("扭率计算测试", test_torsion()))
    results.append(("分支密度测试", test_branching()))
    results.append(("特征提取器测试", test_feature_extractor()))
    
    # 打印结果
    print("\n" + "="*70)
    print("测试结果汇总")
    print("="*70)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✓ 通过" if result else "✗ 失败"
        print(f"{name:20s}: {status}")
    
    print("\n" + "="*70)
    print(f"总计: {passed}/{total} 测试通过")
    print("="*70)
    
    if passed == total:
        print("\n🎉 所有测试通过！系统运行正常。")
        return 0
    else:
        print(f"\n⚠️  有 {total - passed} 个测试失败，请检查错误信息。")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
