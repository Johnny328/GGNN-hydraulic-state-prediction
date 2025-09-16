# -*- coding: utf-8 -*-
"""
生成水力模拟的数据 - 通用版本
@author:
"""
import os
import json
import wntr
import networkx as nx
from scipy.sparse import csr_matrix
import numpy as np
import math

np.random.seed(1)

def load_config(config_path):
    with open(config_path, 'r') as f:
        return json.load(f)

def get_demand_junctions(wn):
    demand_junc = []
    for i in wn.junction_name_list:
        base_demand = wn.get_node(i).demand_timeseries_list[0].base_value
        if base_demand:
            demand_junc.append(i)
    return demand_junc

def create_masks(wn, sensor_names, demand_junc, sensor_sample):
    scenario_names = wn.node_name_list
    jiedian = len(scenario_names)
    mask_measure = np.zeros(jiedian)
    mask_prediction = np.zeros(jiedian)
    mask_sample = np.zeros(jiedian)

    for sen_id in sensor_names:
        if sen_id in scenario_names:
            sensor = scenario_names.index(sen_id)
            mask_measure[sensor] = 1
    for sen_id in demand_junc:
        if sen_id in scenario_names:
            sensor = scenario_names.index(sen_id)
            mask_prediction[sensor] = 1
    for sen_id in sensor_sample:
        if sen_id in scenario_names:
            sensor = scenario_names.index(sen_id)
            mask_sample[sensor] = 1
    return mask_measure, mask_prediction, mask_sample

def set_pressure_bounds(wn, scenario_names):
    jiedian = len(scenario_names)
    pre_bounds_low = np.zeros((jiedian, 2))
    pre_bounds_up = np.zeros((jiedian, 2))
    mask_area = np.zeros(jiedian)

    for i in wn.node_name_list:
        node = scenario_names.index(i)
        if i in ['R1', "R2"]:
            elevation = 100
        else:
            elevation = wn.get_node(i).elevation

        if i in ['n303', 'n336', 'R1', "R2"]:
            upper_bound = 100
            mask_area[node] = 1  # 特殊区域
        elif elevation <= 16:
            upper_bound = 41.11
            mask_area[node] = 2  # Area B
        elif elevation <= 48:
            upper_bound = 75
            mask_area[node] = 3  # Area A
        elif elevation < 100:
            upper_bound = 102.6
            mask_area[node] = 4  # Area C
        pre_bounds_low[node] = [elevation, 1000]
        pre_bounds_up[node] = [elevation, upper_bound]
    return pre_bounds_low, pre_bounds_up, mask_area

def generate_data_for_network(network_config):
    name = network_config['name']
    inp_path = network_config['inp_path']
    sensor_names = network_config['sensor_names']
    sensor_sample = network_config['sensor_sample']

    # 创建输出目录
    output_dir = os.path.join(name, "data_generate")
    os.makedirs(output_dir, exist_ok=True)

    wn = wntr.network.WaterNetworkModel(inp_path)

    # 获取有需求的节点
    demand_junc = get_demand_junctions(wn)

    # 创建掩膜
    mask_measure, mask_prediction, mask_sample = create_masks(wn, sensor_names, demand_junc, sensor_sample)

    #设置压力边界
    scenario_names = wn.node_name_list
    # pre_bounds_low, pre_bounds_up, mask_area = set_pressure_bounds(wn, scenario_names)

    # 模拟参数
    duration = 2
    days = 60
    report_time = 5

    # 重新初始化wn
    wn = wntr.network.WaterNetworkModel(inp_path)
    wn.options.time.report_timestep = 60 * report_time
    wn.options.time.duration = days * 24 * 3600
    # wn.options.hydraulic.demand_model = 'PDD'
    # wn.options.hydraulic.required_pressure = 25
    # wn.options.hydraulic.minimum_pressure = 7
    wn.options.quality.parameter = 'NONE'

    sim = wntr.sim.WNTRSimulator(wn).run_sim()
    sim1=sim.node['head'][scenario_names]
    all_source = np.around(sim1.values, 2)

    # # 数据转换
    # n_frame = int(duration * 60 / report_time + 1)
    # n_slot = all_source.shape[0] - n_frame + 1
    # tmp_seq = np.zeros((n_slot, n_frame, all_source.shape[1]))
    # for i in range(n_slot):
    #     tmp_seq[i, :, :] = all_source[i:i + n_frame, :]
    
    # all_source_1 = all_source[n_frame - 1:]
    # train_len = int(all_source_1.shape[0] * 0.8)
    # seq_train, seq_test = tmp_seq[:train_len], tmp_seq[train_len:]
    # seq_train1, seq_test1 = all_source_1[:train_len], all_source_1[train_len:]
    # print(f'{name} - train: {len(seq_train)}, train series: {len(seq_train1)}')

    # 保存数据
    np.savez_compressed(
        os.path.join(output_dir, "dataset_all_new1.npz"),
        mask_measure=mask_measure,
        mask_prediction=mask_prediction,
        mask_sample=mask_sample,
        dataset=all_source,
    )

def main():
    config = load_config('network_config.json')
    networks = config['networks']
    for network in networks:
        try:
            generate_data_for_network(network)
            print(f"完成网络: {network['name']} 的数据生成。")
        except Exception as e:
            print(f"处理网络 {network['name']} 时出错: {e}")

if __name__ == "__main__":
    main()
