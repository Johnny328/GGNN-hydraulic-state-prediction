# -*- coding: utf-8 -*-
"""
Created on Tue Jul 19 16:18:22 2022

@author: zl489
"""
import sys
import os 
import torch
import networkx as nx
from tqdm import tqdm
import torch.nn as nn
import numpy as np
import errors
import metrics429
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, DataLoader, TensorDataset
np.random.seed(21)

def data_gen(duration,report_time,dataset):
    np.random.seed(21)
    n_frame=int(duration*60/report_time+1)   #historical data + prediction step data        
    n_slot=dataset.shape[0]-n_frame+1
    tmp_seq=np.zeros((n_slot,n_frame,dataset.shape[1]))
    for i in range(n_slot):
        sta=i
        end=sta+n_frame
        tmp_seq[i,:,:]=dataset[sta:end,:]        
    np.random.shuffle(tmp_seq)
    train_len,valid_len=int(dataset.shape[0]*0.6),int(dataset.shape[0]*0.2)
    seq_train,seq_val,seq_test=tmp_seq[:train_len],tmp_seq[train_len:(train_len+valid_len)],tmp_seq[(train_len+valid_len):]
    return seq_train,seq_val,seq_test

def enhance_sensor_impact_np(adj_matrix, sensor_mask, enhancement_factor=10):
    """
    Implement sensor node enhancement of adjacency matrix using NumPy.

    Args:
    adj_matrix (numpy.ndarray): Adjacency matrix, size [num_nodes, num_nodes]
    sensor_indices (list or numpy.ndarray): List of sensor node indices
    enhancement_factor (float): Enhancement factor

    Returns:
    numpy.ndarray: Enhanced adjacency matrix
    """
    # Create a mask with the same shape as the adjacency matrix, initialized to 1
    sensor_indices = np.where(sensor_mask == 1)[0]
    mask = np.ones_like(adj_matrix)

    # Set mask values for rows and columns of sensor nodes
    mask[sensor_indices, :] *= enhancement_factor
    mask[:, sensor_indices] *= enhancement_factor
    # Apply mask to the original adjacency matrix
    enhanced_adj_matrix = adj_matrix * mask
    return enhanced_adj_matrix

def add_self_connections_to_sensors(adj_matrix, sensor_mask, self_weight=1.0):
    """
    Add self-connection weights to sensor nodes in the adjacency matrix.
    
    Parameters:
    adj_matrix (np.array): Original weighted adjacency matrix.
    sensor_mask (np.array): A one-dimensional array where 0 indicates no sensor, 1 indicates a sensor.
    self_weight (float): Weight to be added to self-connections.

    Returns:
    np.array: Modified adjacency matrix.
    """
    # Find indices of sensor positions
    sensor_indices = np.where(sensor_mask == 1)[0]
    # Add weight to diagonal positions for sensor nodes
    adj_matrix[sensor_indices, sensor_indices] += self_weight
    return adj_matrix

def create_graph_from_adjacency_matrix(adj_matrix):
    """Create a directed graph from an adjacency matrix, and add edges and weights"""
    num_nodes = adj_matrix.shape[0]
    print(num_nodes)
    G = nx.DiGraph()
    for i in range(num_nodes):
        for j in range(num_nodes):
            if adj_matrix[i, j] != 0 :  # Assume 0 means no direct connection
                G.add_edge(i, j, weight=adj_matrix[i, j])
    return G
    
def compute_shortest_paths(G, sensor_indices):
    """This is used to calculate the shortest distances from sensor nodes to each node in a directed graph"""
    shortest_paths = {}
    for sensor_idx in sensor_indices:
        # Calculate shortest paths and lengths from each sensor node to all other nodes
        dist = nx.single_source_dijkstra_path_length(G, source=sensor_idx, weight='weight')
        shortest_paths[sensor_idx] = dist
    return shortest_paths

def compute_weight_matrix(shortest_paths, sensor_indices, non_sensor_indices):
    num_sensors = len(sensor_indices)
    num_non_sensors = len(non_sensor_indices)
    min_distances = torch.full((num_sensors, num_non_sensors), float('inf'))
    
    for i, sensor_idx in enumerate(sensor_indices):
        for j, non_sensor_idx in enumerate(non_sensor_indices):
            fwd_dist = shortest_paths['fwd'][sensor_idx].get(non_sensor_idx, float('inf'))
            rev_dist = shortest_paths['rev'][sensor_idx].get(non_sensor_idx, float('inf'))
            min_distances[i, j] = min(fwd_dist, rev_dist)
    
    weights = 1 / (min_distances + 1)  # prevent division-by-zero errors
    weights[min_distances == float('inf')] = 0
    weight_sums = weights.sum(axis=0, keepdim=True)  # Sum across sensor nodes for each non-sensor node
    normalized_weights = weights / weight_sums  # Normalize weights
    # Check whether weights contain nan values
    if torch.isnan(normalized_weights).any():
        print("Warning: weights contains nan values.")
        normalized_weights = torch.nan_to_num(normalized_weights, nan=0.0) 
    
    return normalized_weights.transpose(0, 1)  # transpose so dimensions are correct

def initialize_features_with_min_paths(attr_matrix, shortest_paths_fwd, shortest_paths_rev, sensor_indices, non_sensor_indices):
    # Check attr_matrix dimensions and expand the last dimension if needed
    if attr_matrix.dim() == 2:
        attr_matrix = attr_matrix.unsqueeze(-1)     
    weights = compute_weight_matrix({'fwd': shortest_paths_fwd, 'rev': shortest_paths_rev}, sensor_indices, non_sensor_indices)
    weights_expanded = weights.unsqueeze(0).expand(attr_matrix.size(0), -1, -1)
    
    # Get sensor features
    sensor_features = attr_matrix[:, sensor_indices, :]  # shape [batch_size, num_sensors, feature_size]
    
    # Compute weighted features
    weighted_features = torch.bmm(weights_expanded, sensor_features)  # [batch_size, num_non_sensors, feature_size]    
    # Update features for non-sensor nodes
    attr_matrix[:, non_sensor_indices, :] = weighted_features
    if torch.isnan(attr_matrix).any():
        print("Warning: weights contains nan values.")
    if torch.isinf(attr_matrix).any():
        print("Warning: weights contains inf values.")
    return attr_matrix


def train_model(model, batch_size,feature_size,output_size, epochs,steps_per_epoch,TrainDataset, TestDataset,adj_matrices, dist_matrices,mask_0,mask_1,mask_2,mask_area=None,yanmo=True,percent=0.2,loss_type=0,standard_type=0,threshold=0.0001,init=False,w=1):
    #w is a hyperparameter used to balance the weight between physical loss and prediction accuracy, default is 1
    
       
    NODE_NUM=adj_matrices.shape[0]
    FEATURE_NUM=feature_size
    STEPS_PER_EPOCH = steps_per_epoch

    EPOCHS = epochs  
    BATCH_SIZE = batch_size
    VAL_BATCH_SIZE = min(len(TestDataset),batch_size)

    
    #This mask serves as the input mask (indices of non-sensor nodes)
    mask_measure1=np.argwhere(mask_0==0)[:,0]

    mask_sample=mask_1 
    
    # Standardize node attributes for each region and save as a matrix
    # Create MinMaxScaler object
    region_params = {}
    TrainDataset= torch.tensor(TrainDataset, dtype=torch.float32)
    TestDataset= torch.tensor(TestDataset, dtype=torch.float32)
    
    #This is mainly to add a self-connection node for sensors
    self_attention=False
    if self_attention==True:
        # Call function to add self-connections
        adj_matrices = add_self_connections_to_sensors(adj_matrices, mask_0, self_weight=1.0)
    
    if standard_type==0:
        pass
    elif standard_type == 1:        
        # Calculate mean and standard deviation for all regions
        mean_value = torch.mean(TrainDataset)
        std_value = torch.std(TrainDataset)

        # Apply z-score normalization to training and test sets
        TrainDataset = (TrainDataset - mean_value) / std_value
        TestDataset = (TestDataset - mean_value) / std_value
        # Save mean and standard deviation
        region_params = {'mean': mean_value, 'std': std_value}
    elif standard_type == 2:
        # Perform max-min normalization for different regions
        for region in range(1, 5):  # Region numbers from 1 to 4
            region_indices = torch.nonzero(mask_area == region).squeeze()  # Get indices of nodes belonging to current region            
            mean_value = torch.mean(TrainDataset[:, :, region_indices])
            std_value = torch.std(TrainDataset[:, :, region_indices])            
            # Apply z-score normalization to training and test sets
            TrainDataset[:, :, region_indices] = (TrainDataset[:, :, region_indices] - mean_value) / std_value
            TestDataset[:, :, region_indices] = (TestDataset[:, :, region_indices] - mean_value) / std_value

            region_params[region] = {'mean': mean_value, 'std': std_value}
            
    # Split train dataset in batches
    train_data = DataLoader(TrainDataset, batch_size=BATCH_SIZE, shuffle=True)
    val_data = DataLoader(TestDataset, batch_size=VAL_BATCH_SIZE, shuffle=False) 
        
    #criterion = nn.NLLLoss()
    # When using regression, MSE (Mean Squared Error) loss is often used
    # criterion = nn.MSELoss()
    # When using regression, MAE (Mean Absolute Error / L1) loss can be used
    # criterion = nn.L1Loss()  

    optimizer = torch.optim.Adam(model.parameters())        
    # Initial evaluations
    # =========================================================================    
    loss_ls = []
    mae_ls,mse_ls,rmse_ls,mape_ls,mspe_ls = [],[],[],[],[]


    
    #adj_matrices=enhance_sensor_impact_np(adj_matrices, mask_0, enhancement_factor=10)
    
    
    G = create_graph_from_adjacency_matrix(dist_matrices)
    sensor_indices=np.argwhere(mask_0==1)[:,0]        
    # Compute forward shortest paths
    shortest_paths_fwd = compute_shortest_paths(G, sensor_indices)

    # Compute reverse shortest paths
    G_rev = G.reverse()
    shortest_paths_rev = compute_shortest_paths(G_rev, sensor_indices)
    a = adj_matrices
    # d = dist_matrices
    A_base = torch.from_numpy(a).float().cuda()  
    # D_base = torch.from_numpy(d).float().cuda()  

    # Training 
    # =========================================================================
    
    for epoch in range(EPOCHS):

        # Break condition
        #if (uWBE_mean_ls[-1] < 0.2):
        #    break
        epoch_loss = []
        with tqdm(total=STEPS_PER_EPOCH, file=sys.stdout) as pbar:
            model.train()
            for step in range(STEPS_PER_EPOCH):
                
                
                train_input = next(iter(train_data)).float().swapaxes(1,2)
                #T=train_input[:,:,-1]                
                # train_input= initialize_features_with_min_paths(train_input, shortest_paths_fwd, shortest_paths_rev, sensor_indices, non_sensor_indices)
               
                
                if yanmo and percent>0:
                    #Need to copy mask_0 here, as later tests need to use all data for input, modifying it would affect subsequent test results
                    #If masking is true, then randomly select 20% of the sensors' values to set to 0 during training
                    mask_00=mask_0.copy()
                    
                    #indices=np.where(mask_00 == 1)[0]
                    #Now updating the mask sites: selecting masked nodes from sensor nodes in mask_sample, but still applying the mask in mask_0
                    indices=np.where(mask_sample == 1)[0]
                    num_to_select = int(len(indices)*percent)
                    if num_to_select:
                        selected_indices = np.random.choice(indices, size=num_to_select, replace=True)
                        mask_00[selected_indices] = 0
                    #index = np.argwhere(mask_00==0)
                    mask_measure_1=np.argwhere(mask_00==0)[:,0]
                    train_input[:,mask_measure_1,:]=0
                
                elif init:
                    sensor_indices=np.argwhere(mask_0==1)[:,0]
                    non_sensor_indices=np.argwhere(mask_0==0)[:,0]
                    train_input= initialize_features_with_min_paths(train_input, shortest_paths_fwd, shortest_paths_rev, sensor_indices, non_sensor_indices)    
                else:
                    train_input[:,mask_measure1,:]=0      
                   
                X=train_input[:,:,:-output_size]
                #print(X.shape)
                T=train_input[:,:,-output_size:]   
                    
                # else:
                #     #Need to note that if masking operation is performed, the mask will change each time, so the test set mask later needs to be ensured to be the same as at the beginning
                #     mask_measure_1= np.argwhere(mask_0==0)[:,0]  
                #     X[:,mask_measure_1,:]=0


                
 
                                         
                # Adjust the shapes of A and D to match the batch size
                A = A_base.expand(X.shape[0], -1, -1)  # Use expand to save memory
                Y = model(X.cuda(),A)
                
                if standard_type==0:
                    pass
                elif standard_type == 1:
                    # # Unified max-min normalization for all regions
                    # Y = Y * (region_params['max']-region_params['min']) + min_value
                    # T = T * (region_params['max']-region_params['min']) + min_value
                    
                    # Unified z-score normalization for all regions
                    mean_value = region_params['mean']
                    std_value = region_params['std']
                    Y = Y*std_value + mean_value
                    T = T*std_value + mean_value

                elif standard_type == 2:
                    # Max-min normalization by different regions
                    for region in range(1, 5):  # Region numbers from 1 to 4
                        region_indices = torch.nonzero(mask_area == region).squeeze()  # Get indices of nodes belonging to current region
                        # min_value = region_params[region]['min']
                        # max_value = region_params[region]['max']
                        # Y[:, region_indices] = Y[:, region_indices] * (max_value - min_value) + min_value
                        # T[:, region_indices] = T[:, region_indices] * (max_value - min_value) + min_value 
                        mean_value = region_params[region]['mean']
                        std_value = region_params[region]['std']
                        Y[:, region_indices] = Y[:, region_indices]*std_value + mean_value
                        T[:, region_indices] = T[:, region_indices]*std_value + mean_value
                
                loss = metrics429.custom_loss_extend_physic(Y, T,A, mask_0, mask_2,loss_type,threshold,w)
                # Zero gradients, perform a backward pass, and update the weights.
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                pbar.set_description('...step %d/%d - loss: %.4f' % ((step+1), STEPS_PER_EPOCH, loss.item()))
                pbar.update(1)
                epoch_loss.append(loss.item())
                loss_ls.append(loss.item())

                
        # =====================================================================
        
        # New evaluations
        # =====================================================================        

        # Test dataset
        # ---------------------------------------------------------------------
        model.eval()
        num_sum=0
        Y_all=[]
        T_all=[]
        for batch in val_data:
            num_sum+=len(batch)
            batch=batch.float().swapaxes(1,2)
            X=batch[:,:,:-output_size]
            
            X[:,mask_measure1,:]=0
            
            # Add initial feature attributes
            if percent==0 and init:
                sensor_indices=np.argwhere(mask_0==1)[:,0]
                non_sensor_indices=np.argwhere(mask_0==0)[:,0]                
                X = initialize_features_with_min_paths(X, shortest_paths_fwd, shortest_paths_rev, sensor_indices, non_sensor_indices)
            
            T=batch[:,:,-output_size:]
                           
            # Adjust the shapes of A and D to match the batch size
            A = A_base.expand(X.shape[0], -1, -1)  # Use expand to save memory
            Y = model(X.cuda(),A) 
            
            if standard_type==0:
                pass
            elif standard_type == 1:
                
                # Apply z-score normalization uniformly across all regions
                mean_value = region_params['mean']
                std_value = region_params['std']
                Y = Y*std_value + mean_value
                T = T*std_value + mean_value

            elif standard_type == 2:
                # Apply region-wise normalization (z-score using stored region params)
                for region in range(1, 5):  # Region numbers from 1 to 4
                    region_indices = torch.nonzero(mask_area == region).squeeze()  # Get indices of nodes belonging to current region                    
                    mean_value = region_params[region]['mean']
                    std_value = region_params[region]['std']
                    Y[:, region_indices] = Y[:, region_indices]*std_value + mean_value
                    T[:, region_indices] = T[:, region_indices]*std_value + mean_value   
                        
            Y__=Y.detach().cpu().numpy()
            T__=T.detach().cpu().numpy()
                        
            nonzero_cols1 = np.argwhere(mask_2==1)[:,0]
            #Select columns that are not all zeros to form new tensors, and only compare the last value
            Y_=Y__[:,nonzero_cols1,-1]
            T_=T__[:,nonzero_cols1,-1]
            
            Y_all.append(Y_)
            T_all.append(T_)
            del X, Y, A
        
        # Y_all=np.concatenate(Y_,axis=0)  #[B,L,D]-> [N, L, D]
        # T_all=np.concatenate(T_,axis=0)
        Y_all=np.concatenate(Y_all,axis=0)  #[B,L,D]-> [N, L, D]
        T_all=np.concatenate(T_all,axis=0)
        mae, mse, rmse, mape, mspe = metrics429.metric(Y_all, T_all) 
        mae_ls.append(mae)
        mse_ls.append(mse)
        rmse_ls.append(rmse)
        mape_ls.append(mape)
        mspe_ls.append(mspe)


        # Print results
        # ---------------------------------------------------------------------

        print('Epoch %d/%d - train_loss: %.4f / %.4f / %.4f (Min/Avg/Max)'  # - val_loss: %.4f'
            % (epoch+1, EPOCHS, min(epoch_loss), sum(epoch_loss)/len(epoch_loss), max(epoch_loss)))  #, val_loss))
            
        print('\t      ---------------------------------------------------------------------' )
        print('[Average value] mae:{}, rmse:{}, mape:{}'.format(mae_ls[-1], rmse_ls[-1],mape_ls[-1]))

        print('\t      ---------------------------------------------------------------------')
    # =========================================================================
    statistics = {'loss_ls': loss_ls,
                    'mae':mae_ls,
                    'mse':mse_ls,
                    'rmse':rmse_ls,
                    'mape':mape_ls,
                    'mspe':mspe_ls}

    return model, optimizer, statistics
