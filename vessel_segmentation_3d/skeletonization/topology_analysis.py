"""
血管拓扑结构分析
================

分析血管骨架的拓扑结构，包括：
- 分叉点识别
- 端点识别
- 分支提取
- 拓扑图构建

作者：医学影像AI研究团队
日期：2026-04-08
"""

import numpy as np
from typing import Tuple, List, Dict, Optional
from scipy.spatial import KDTree
import networkx as nx
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def analyze_vessel_topology(skeleton_points: np.ndarray,
                            connectivity: int = 26) -> Tuple[nx.Graph, List, List, List]:
    """
    分析血管拓扑结构
    
    构建血管网络的拓扑图，识别关键结构。
    
    步骤：
    1. 构建KD树加速邻域搜索
    2. 创建图结构
    3. 识别分叉点（度>2）
    4. 识别端点（度=1）
    5. 提取分支路径
    
    参数:
        skeleton_points (np.ndarray): 骨架点坐标 (N, 3)
        connectivity (int): 连通性（6, 18, 或 26邻域）
    
    返回:
        graph (nx.Graph): 血管网拓扑图
        branches (List): 分支列表，每个分支是点索引列表
        junctions (List): 分叉点索引列表
        endpoints (List): 端点索引列表
    
    示例:
        >>> skeleton_points = np.array([[10, 10, 10], [11, 10, 10], [12, 10, 10]])
        >>> graph, branches, junctions, endpoints = analyze_vessel_topology(skeleton_points)
        >>> print(f"分支数量: {len(branches)}")
    """
    logger.info("开始拓扑结构分析...")
    
    # 根据连通性确定搜索半径
    if connectivity == 6:
        radius = 1.1
    elif connectivity == 18:
        radius = 1.5
    else:  # 26邻域
        radius = 1.8
    
    # 创建KD树用于快速邻域搜索
    tree = KDTree(skeleton_points)
    
    # 创建图
    graph = nx.Graph()
    
    # 添加节点
    for i, point in enumerate(skeleton_points):
        graph.add_node(i, position=point)
    
    # 添加边（连接相邻的骨架点）
    logger.info("构建图结构...")
    for i, point in enumerate(skeleton_points):
        # 查找邻居
        neighbors = tree.query_ball_point(point, r=radius)
        
        for j in neighbors:
            if i != j:
                graph.add_edge(i, j)
    
    # 识别分叉点（度>2的节点）
    junctions = [node for node, degree in graph.degree() if degree > 2]
    logger.info(f"识别到 {len(junctions)} 个分叉点")
    
    # 识别端点（度=1的节点）
    endpoints = [node for node, degree in graph.degree() if degree == 1]
    logger.info(f"识别到 {len(endpoints)} 个端点")
    
    # 提取分支（两个分叉点之间的路径）
    logger.info("提取分支...")
    branches = extract_branches(graph, junctions, endpoints)
    logger.info(f"提取到 {len(branches)} 个分支")
    
    # 统计信息
    logger.info(f"总节点数: {graph.number_of_nodes()}")
    logger.info(f"总边数: {graph.number_of_edges()}")
    
    return graph, branches, junctions, endpoints


def extract_branches(graph: nx.Graph,
                     junctions: List[int],
                     endpoints: List[int]) -> List[List[int]]:
    """
    提取血管分支
    
    从分叉点和端点开始追踪分支路径。
    
    参数:
        graph: 血管网图
        junctions: 分叉点列表
        endpoints: 端点列表
    
    返回:
        branches: 分支列表
    """
    branches = []
    visited_edges = set()
    
    # 从每个分叉点开始追踪
    for junction in junctions:
        neighbors = list(graph.neighbors(junction))
        
        for neighbor in neighbors:
            # 检查边是否已访问
            edge = tuple(sorted([junction, neighbor]))
            if edge in visited_edges:
                continue
            
            # 追踪分支
            branch = trace_branch(graph, junction, neighbor, junctions, endpoints)
            
            if len(branch) > 1:
                branches.append(branch)
                
                # 标记边为已访问
                for i in range(len(branch) - 1):
                    edge = tuple(sorted([branch[i], branch[i+1]]))
                    visited_edges.add(edge)
    
    # 从端点开始追踪（处理孤立分支）
    for endpoint in endpoints:
        neighbors = list(graph.neighbors(endpoint))
        
        for neighbor in neighbors:
            edge = tuple(sorted([endpoint, neighbor]))
            if edge in visited_edges:
                continue
            
            branch = trace_branch(graph, endpoint, neighbor, junctions, endpoints)
            
            if len(branch) > 1:
                branches.append(branch)
    
    return branches


def trace_branch(graph: nx.Graph,
                 start: int,
                 next_node: int,
                 junctions: List[int],
                 endpoints: List[int]) -> List[int]:
    """
    追踪单个分支
    
    从起始点开始，沿着血管追踪直到遇到分叉点或端点。
    
    参数:
        graph: 血管网图
        start: 起始节点
        next_node: 下一个节点
        junctions: 分叉点列表
        endpoints: 端点列表
    
    返回:
        branch: 分支节点列表
    """
    branch = [start, next_node]
    current = next_node
    previous = start
    
    # 最大追踪长度（防止无限循环）
    max_length = 10000
    
    while len(branch) < max_length:
        neighbors = list(graph.neighbors(current))
        
        # 如果到达分叉点或端点，停止
        if current in junctions or current in endpoints:
            break
        
        # 找到下一个节点（不是前一个节点）
        next_nodes = [n for n in neighbors if n != previous]
        
        if len(next_nodes) == 0:
            # 死胡同
            break
        
        if len(next_nodes) > 1:
            # 遇到新的分叉点
            break
        
        next_node = next_nodes[0]
        branch.append(next_node)
        previous = current
        current = next_node
    
    return branch


def calculate_branch_lengths(graph: nx.Graph,
                             branches: List[List[int]],
                             spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0)) -> np.ndarray:
    """
    计算分支长度
    
    参数:
        graph: 血管网图
        branches: 分支列表
        spacing: 体素间距 (mm)
    
    返回:
        lengths: 分支长度数组 (mm)
    """
    lengths = []
    
    for branch in branches:
        if len(branch) < 2:
            lengths.append(0.0)
            continue
        
        # 获取分支点坐标
        positions = [graph.nodes[node]['position'] for node in branch]
        positions = np.array(positions)
        
        # 计算相邻点之间的距离
        distances = np.sqrt(np.sum(np.diff(positions, axis=0)**2 * np.array(spacing)**2, axis=1))
        
        # 总长度
        total_length = np.sum(distances)
        lengths.append(total_length)
    
    return np.array(lengths)


def calculate_branch_angles(graph: nx.Graph,
                            junctions: List[int]) -> Dict[int, List[float]]:
    """
    计算分叉点处的分支角度
    
    参数:
        graph: 血管网图
        junctions: 分叉点列表
    
    返回:
        angles: 每个分叉点的角度列表
    """
    angles = {}
    
    for junction in junctions:
        neighbors = list(graph.neighbors(junction))
        
        if len(neighbors) < 2:
            angles[junction] = []
            continue
        
        # 获取分叉点和邻居的坐标
        junction_pos = graph.nodes[junction]['position']
        
        # 计算每个分支的方向向量
        directions = []
        for neighbor in neighbors:
            neighbor_pos = graph.nodes[neighbor]['position']
            direction = np.array(neighbor_pos) - np.array(junction_pos)
            direction = direction / (np.linalg.norm(direction) + 1e-8)
            directions.append(direction)
        
        # 计算所有分支对之间的角度
        junction_angles = []
        for i in range(len(directions)):
            for j in range(i + 1, len(directions)):
                angle = np.arccos(np.clip(np.dot(directions[i], directions[j]), -1.0, 1.0))
                angle_deg = np.degrees(angle)
                junction_angles.append(angle_deg)
        
        angles[junction] = junction_angles
    
    return angles


def get_topology_statistics(graph: nx.Graph,
                            branches: List[List[int]],
                            junctions: List[int],
                            endpoints: List[int]) -> Dict:
    """
    获取拓扑统计信息
    
    参数:
        graph: 血管网图
        branches: 分支列表
        junctions: 分叉点列表
        endpoints: 端点列表
    
    返回:
        stats: 统计信息字典
    """
    stats = {
        'total_nodes': graph.number_of_nodes(),
        'total_edges': graph.number_of_edges(),
        'num_junctions': len(junctions),
        'num_endpoints': len(endpoints),
        'num_branches': len(branches),
        'avg_branch_length': 0.0,
        'max_branch_length': 0.0,
        'avg_node_degree': 0.0,
        'max_node_degree': 0,
    }
    
    # 分支长度统计
    if len(branches) > 0:
        lengths = calculate_branch_lengths(graph, branches)
        stats['avg_branch_length'] = float(np.mean(lengths))
        stats['max_branch_length'] = float(np.max(lengths))
    
    # 节点度统计
    degrees = [degree for node, degree in graph.degree()]
    if len(degrees) > 0:
        stats['avg_node_degree'] = float(np.mean(degrees))
        stats['max_node_degree'] = int(np.max(degrees))
    
    return stats


# 测试代码
if __name__ == "__main__":
    print("="*60)
    print("测试拓扑结构分析")
    print("="*60)
    
    # 创建测试骨架（Y形结构）
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
    graph, branches, junctions, endpoints = analyze_vessel_topology(skeleton_points)
    
    print(f"\n分叉点数量: {len(junctions)}")
    print(f"端点数量: {len(endpoints)}")
    print(f"分支数量: {len(branches)}")
    
    # 获取统计信息
    stats = get_topology_statistics(graph, branches, junctions, endpoints)
    
    print("\n拓扑统计:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # 计算分支角度
    angles = calculate_branch_angles(graph, junctions)
    
    print("\n分叉角度:")
    for junction, junction_angles in angles.items():
        print(f"  分叉点 {junction}: {junction_angles}")
    
    print("\n✓ 测试完成!")
