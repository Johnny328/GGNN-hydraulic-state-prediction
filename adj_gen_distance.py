# -*- coding: utf-8 -*-
"""
Generate adjacency matrices for hydraulic simulation - general version
@author:
"""
import os
import json
import wntr
import networkx as nx
from scipy.sparse import csr_matrix
import numpy as np
import pandas as pd

# Set random seed
np.random.seed(1)

def load_config(config_path):
    """
    Load configuration file.
    """
    with open(config_path, 'r') as f:
        return json.load(f)


def compute_dis_weight(wn, method, num_special_links):
    """
    Compute weights for the network based on different methods.

    Parameters:
    - wn: WaterNetworkModel object
    - method: Weight computation method, string type, can be 'length', 'diameter_length', 'binary'
    - num_special_links: Integer, number of special links to exclude (such as pumps and valves)

    Returns:
    - normalized_weight: numpy array, normalized weight array
    """
    weight = []
    
    # Adjust slicing to include all links when num_special_links is 0
    if num_special_links > 0:
        link_names = wn.link_name_list[:-num_special_links]  # Exclude pumps and valves
    else:
        link_names = wn.link_name_list  # Include all links

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

    # Consider weights for pumps and valves as 0
    if num_special_links > 0:
        normalized_weight = np.append(normalized_weight, [0] * num_special_links)

    return normalized_weight

def compute_weight(wn, method, num_special_links):
    """
    Compute weights for the network based on different methods.

    Parameters:
    - wn: WaterNetworkModel object
    - method: Weight computation method, string type, can be 'length', 'diameter_length', 'binary'
    - num_special_links: Integer, number of special links to exclude (such as pumps and valves)

    Returns:
    - normalized_weight: numpy array, normalized weight array
    """
    weight = []
    
    # Adjust slicing to include all links when num_special_links is 0
    if num_special_links > 0:
        link_names = wn.link_name_list[:-num_special_links]  # Exclude pumps and valves
    else:
        link_names = wn.link_name_list  # Include all links

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

    # Consider weights for pumps and valves as 0
    if num_special_links > 0:
        normalized_weight = np.append(normalized_weight, [0] * num_special_links)

    return normalized_weight


def generate_adj_matrices(wn, flowrate_all_times, weight_method, is_undirected, num_special_links):
    """
    Generate adjacency matrices for all times.
    """
    normalized_weight = compute_weight(wn, weight_method, num_special_links)
    normalized_weight_series = pd.Series(normalized_weight, index=flowrate_all_times.columns, dtype=flowrate_all_times.dtypes[0])

    num_times = flowrate_all_times.shape[0]
    num_nodes = len(wn.node_name_list)
    # Initialize combined adjacency matrix as zero matrix
    combined_adj = np.zeros((num_nodes, num_nodes))

    for t in range(num_times):
        flowrate = flowrate_all_times.iloc[t, :].copy()
        flowrate_sign = np.sign(flowrate)
        # Calculate weighted flowrate
        weighted_flowrate = flowrate_sign * normalized_weight_series

        # Build graph
        G1 = wn.to_graph(link_weight=weighted_flowrate, modify_direction=True)

        if is_undirected:
            G1 = G1.to_undirected()

        # Get adjacency matrix as dense array
        node_adj = nx.adjacency_matrix(G1).toarray()

        # Element-wise maximum update
        combined_adj = np.maximum(combined_adj, node_adj)

    # # Convert to sparse matrix format
    # combined_adj_sparse = csr_matrix(combined_adj)
    #return combined_adj_sparse
    return combined_adj

def generate_adj_dis_matrices(wn, flowrate_all_times, weight_method, is_undirected, num_special_links):
    """
    Generate adjacency matrices for all times.
    """
    normalized_weight = compute_dis_weight(wn, weight_method, num_special_links)
    normalized_weight_series = pd.Series(normalized_weight, index=flowrate_all_times.columns, dtype=flowrate_all_times.dtypes[0])

    num_times = flowrate_all_times.shape[0]
    num_nodes = len(wn.node_name_list)
    # Initialize combined adjacency matrix as zero matrix
    combined_adj = np.zeros((num_nodes, num_nodes))

    for t in range(num_times):
        flowrate = flowrate_all_times.iloc[t, :].copy()
        flowrate_sign = np.sign(flowrate)
        # Calculate weighted flowrate
        weighted_flowrate = flowrate_sign * normalized_weight_series

        # Build graph
        G1 = wn.to_graph(link_weight=weighted_flowrate, modify_direction=True)

        if is_undirected:
            G1 = G1.to_undirected()

        # Get adjacency matrix as dense array
        node_adj = nx.adjacency_matrix(G1).toarray()

        # Element-wise maximum update
        combined_adj = np.maximum(combined_adj, node_adj)

    # # Convert to sparse matrix format
    # combined_adj_sparse = csr_matrix(combined_adj)
    #return combined_adj_sparse
    return combined_adj

def generate_adjacency_matrices_for_network(network_config,distance=True):
    """
    Generate adjacency matrices for a single network and save them.
    """
    name = network_config['name']
    inp_path = network_config['inp_path']
    weight_method = network_config.get('weight_method', 'length')  # Default to 'length'
    is_undirected = network_config.get('is_undirected', False)    # Default to directed graph

    # Create output directory
    output_dir = os.path.join(name, "data_generate")
    os.makedirs(output_dir, exist_ok=True)

    # Load water network model
    wn = wntr.network.WaterNetworkModel(inp_path)
    wn.options.time.report_timestep = 60 * 5  # report_time=5
    wn.options.time.duration = 1 * 24 * 3600  # days=1
    wn.options.hydraulic.demand_model = 'PDD'
    wn.options.hydraulic.required_pressure = 25
    wn.options.hydraulic.minimum_pressure = 7

    # Run simulation
    sim = wntr.sim.EpanetSimulator(wn)
    results = sim.run_sim()

    # Get flowrate data for all times
    flowrate_all_times = results.link['flowrate']

    # Get number of pumps and valves
    num_pumps = len(wn.pump_name_list)
    num_valves = len(wn.valve_name_list)
    num_special_links = num_pumps + num_valves
    print(f"Network: {name}, Pumps: {num_pumps}, Valves: {num_valves}, Special Links: {num_special_links}")
    
    # Save adjacency matrices
    file_suffix = "undirected" if is_undirected else "directed"

    # Generate adjacency matrices
    if distance:        
        adj_matrices = generate_adj_dis_matrices(wn, flowrate_all_times, weight_method, is_undirected, num_special_links)
        adj_filename = f"adj_matrices_link_distance_{weight_method}_{file_suffix}.npy"
    else:
        adj_matrices = generate_adj_matrices(wn, flowrate_all_times, weight_method, is_undirected, num_special_links)
        adj_filename = f"adj_matrices_link_weight_{weight_method}_{file_suffix}.npy"
    # Check for NaN values
    for i, adj in enumerate(adj_matrices):
        if np.isnan(adj.data).any():
            print(f"NaN value found in adjacency matrix at time step {i} for network {name}")

    adj_filepath = os.path.join(output_dir, adj_filename)
    np.save(adj_filepath, adj_matrices)
    print(f"Adjacency matrices saved to {adj_filepath}")

def main():
    # Load configuration file
    config = load_config('adj_config.json')
    networks = config.get('networks', [])

    if not networks:
        print("No network configurations found in config.json.")
        return

    # Iterate through all networks and generate adjacency matrices
    for network in networks:
        try:
            generate_adjacency_matrices_for_network(network,distance=False)
            generate_adjacency_matrices_for_network(network,distance=True)
            print(f"Completed processing for network: {network['name']}")
        except Exception as e:
            print(f"Error processing network {network.get('name', 'Unknown')}: {e}")

if __name__ == "__main__":
    main()
