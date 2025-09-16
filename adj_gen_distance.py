# -*- coding: utf-8 -*-
"""
生成水力模拟的邻接矩阵 - 通用版本
@author:
"""
import os
import json
import wntr
import networkx as nx
from scipy.sparse import csr_matrix
import numpy as np
import pandas as pd

# 设置随机种子
np.random.seed(1)

def load_config(config_path):
    """
    加载配置文件。
    """
    with open(config_path, 'r') as f:
        return json.load(f)


def compute_dis_weight(wn, method, num_special_links):
    """
    计算管网的权重，根据不同的方法。

    参数:
    - wn: WaterNetworkModel 对象
    - method: 权重计算方法，字符串类型，可以是 'length', 'diameter_length', 'binary'
    - num_special_links: 整数，表示需要排除的特殊链接数量（如泵和阀门）

    返回:
    - normalized_weight: numpy 数组，归一化后的权重数组
    """
    weight = []
    
    # 调整切片操作，确保当 num_special_links 为 0 时包含所有链接
    if num_special_links > 0:
        link_names = wn.link_name_list[:-num_special_links]  # 排除泵和阀门
    else:
        link_names = wn.link_name_list  # 包含所有链接

    if method == 'length':
        for link in link_names:
            L = wn.get_link(link).length
            weight.append(L if L != 0 else 0)
    elif method == 'diameter_length':
        for link in link_names:
            L = wn.get_link(link).length
            D = wn.get_link(link).diameter
            w = L/(D ** 1.166) if L != 0 else 0
            weight.append(w)
    elif method == 'binary':
        weight = [1 for _ in link_names]
    else:
        raise ValueError(f"Unsupported weight computation method: {method}")

    weight = np.array(weight)

    if method != 'binary':
        if weight.size == 0:
            normalized_weight = np.array([])
        else:
            weight_max, weight_min = weight.max(), weight.min()
            # b = 0.001 * (weight_max - weight_min)
            # normalized_weight = (weight - weight_min + b) / ((weight_max - weight_min) + b)
            normalized_weight = weight /weight_max
    else:
        normalized_weight = np.array(weight)

    # 考虑泵和阀门的权重为0
    if num_special_links > 0:
        normalized_weight = np.append(normalized_weight, [0] * num_special_links)

    return normalized_weight

def compute_weight(wn, method, num_special_links):
    """
    计算管网的权重，根据不同的方法。

    参数:
    - wn: WaterNetworkModel 对象
    - method: 权重计算方法，字符串类型，可以是 'length', 'diameter_length', 'binary'
    - num_special_links: 整数，表示需要排除的特殊链接数量（如泵和阀门）

    返回:
    - normalized_weight: numpy 数组，归一化后的权重数组
    """
    weight = []
    
    # 调整切片操作，确保当 num_special_links 为 0 时包含所有链接
    if num_special_links > 0:
        link_names = wn.link_name_list[:-num_special_links]  # 排除泵和阀门
    else:
        link_names = wn.link_name_list  # 包含所有链接

    if method == 'length':
        for link in link_names:
            L = wn.get_link(link).length
            weight.append(1 / L if L != 0 else 0)
    elif method == 'diameter_length':
        for link in link_names:
            L = wn.get_link(link).length
            D = wn.get_link(link).diameter
            w = (D ** 1.166) / L if L != 0 else 0
            weight.append(w)
    elif method == 'binary':
        weight = [1 for _ in link_names]
    else:
        raise ValueError(f"Unsupported weight computation method: {method}")

    weight = np.array(weight)

    if method != 'binary':
        if weight.size == 0:
            normalized_weight = np.array([])
        else:
            weight_max, weight_min = weight.max(), weight.min()
            # b = 0.001 * (weight_max - weight_min)
            # normalized_weight = (weight - weight_min + b) / ((weight_max - weight_min) + b)
            normalized_weight = weight /weight_max
    else:
        normalized_weight = np.array(weight)

    # 考虑泵和阀门的权重为0
    if num_special_links > 0:
        normalized_weight = np.append(normalized_weight, [0] * num_special_links)

    return normalized_weight


def generate_adj_matrices(wn, flowrate_all_times, weight_method, is_undirected, num_special_links):
    """
    生成所有时刻的邻接矩阵。
    """
    normalized_weight = compute_weight(wn, weight_method, num_special_links)
    normalized_weight_series = pd.Series(normalized_weight, index=flowrate_all_times.columns, dtype=flowrate_all_times.dtypes[0])

    num_times = flowrate_all_times.shape[0]
    num_nodes = len(wn.node_name_list)
    # 初始化综合邻接矩阵为全零矩阵
    combined_adj = np.zeros((num_nodes, num_nodes))

    for t in range(num_times):
        flowrate = flowrate_all_times.iloc[t, :].copy()
        flowrate_sign = np.sign(flowrate)
        # 计算加权流量
        weighted_flowrate = flowrate_sign * normalized_weight_series

        # 构建图
        G1 = wn.to_graph(link_weight=weighted_flowrate, modify_direction=True)

        if is_undirected:
            G1 = G1.to_undirected()

        # 获取邻接矩阵为密集数组
        node_adj = nx.adjacency_matrix(G1).toarray()

        # 元素级最大值更新
        combined_adj = np.maximum(combined_adj, node_adj)

    # # 转换为稀疏矩阵格式
    # combined_adj_sparse = csr_matrix(combined_adj)
    #return combined_adj_sparse
    return combined_adj

def generate_adj_dis_matrices(wn, flowrate_all_times, weight_method, is_undirected, num_special_links):
    """
    生成所有时刻的邻接矩阵。
    """
    normalized_weight = compute_dis_weight(wn, weight_method, num_special_links)
    normalized_weight_series = pd.Series(normalized_weight, index=flowrate_all_times.columns, dtype=flowrate_all_times.dtypes[0])

    num_times = flowrate_all_times.shape[0]
    num_nodes = len(wn.node_name_list)
    # 初始化综合邻接矩阵为全零矩阵
    combined_adj = np.zeros((num_nodes, num_nodes))

    for t in range(num_times):
        flowrate = flowrate_all_times.iloc[t, :].copy()
        flowrate_sign = np.sign(flowrate)
        # 计算加权流量
        weighted_flowrate = flowrate_sign * normalized_weight_series

        # 构建图
        G1 = wn.to_graph(link_weight=weighted_flowrate, modify_direction=True)

        if is_undirected:
            G1 = G1.to_undirected()

        # 获取邻接矩阵为密集数组
        node_adj = nx.adjacency_matrix(G1).toarray()

        # 元素级最大值更新
        combined_adj = np.maximum(combined_adj, node_adj)

    # # 转换为稀疏矩阵格式
    # combined_adj_sparse = csr_matrix(combined_adj)
    #return combined_adj_sparse
    return combined_adj

def generate_adjacency_matrices_for_network(network_config,distance=True):
    """
    为单个网络生成邻接矩阵并保存。
    """
    name = network_config['name']
    inp_path = network_config['inp_path']
    weight_method = network_config.get('weight_method', 'length')  # 默认使用 'length'
    is_undirected = network_config.get('is_undirected', False)    # 默认生成有向图

    # 创建输出目录
    output_dir = os.path.join(name, "data_generate")
    os.makedirs(output_dir, exist_ok=True)

    # 加载水网络模型
    wn = wntr.network.WaterNetworkModel(inp_path)
    wn.options.time.report_timestep = 60 * 5  # report_time=5
    wn.options.time.duration = 1 * 24 * 3600  # days=1
    wn.options.hydraulic.demand_model = 'PDD'
    wn.options.hydraulic.required_pressure = 25
    wn.options.hydraulic.minimum_pressure = 7

    # 运行仿真
    sim = wntr.sim.EpanetSimulator(wn)
    results = sim.run_sim()

    # 获取所有时刻的流量数据
    flowrate_all_times = results.link['flowrate']

    # 获取泵和阀门的数量
    num_pumps = len(wn.pump_name_list)
    num_valves = len(wn.valve_name_list)
    num_special_links = num_pumps + num_valves
    print(f"Network: {name}, Pumps: {num_pumps}, Valves: {num_valves}, Special Links: {num_special_links}")
    
    # 保存邻接矩阵
    file_suffix = "undirected" if is_undirected else "directed"

    # 生成邻接矩阵
    if distance:        
        adj_matrices = generate_adj_dis_matrices(wn, flowrate_all_times, weight_method, is_undirected, num_special_links)
        adj_filename = f"adj_matrices_link_distance_{weight_method}_{file_suffix}.npy"
    else:
        adj_matrices = generate_adj_matrices(wn, flowrate_all_times, weight_method, is_undirected, num_special_links)
        adj_filename = f"adj_matrices_link_weight_{weight_method}_{file_suffix}.npy"
    # 检查是否存在 NaN 值
    for i, adj in enumerate(adj_matrices):
        if np.isnan(adj.data).any():
            print(f"NaN value found in adjacency matrix at time step {i} for network {name}")

    adj_filepath = os.path.join(output_dir, adj_filename)
    np.save(adj_filepath, adj_matrices)
    print(f"Adjacency matrices saved to {adj_filepath}")

def main():
    # 加载配置文件
    config = load_config('adj_config.json')
    networks = config.get('networks', [])

    if not networks:
        print("No network configurations found in config.json.")
        return

    # 遍历所有网络并生成邻接矩阵
    for network in networks:
        try:
            generate_adjacency_matrices_for_network(network,distance=False)
            generate_adjacency_matrices_for_network(network,distance=True)
            print(f"Completed processing for network: {network['name']}")
        except Exception as e:
            print(f"Error processing network {network.get('name', 'Unknown')}: {e}")

if __name__ == "__main__":
    main()
