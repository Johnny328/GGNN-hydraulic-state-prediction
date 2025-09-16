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
    n_frame=int(duration*60/report_time+1)   #历史数据+预测步长数据        
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
    使用 NumPy 实现邻接矩阵的传感器节点增强。

    Args:
    adj_matrix (numpy.ndarray): 邻接矩阵，大小为 [num_nodes, num_nodes]
    sensor_indices (list or numpy.ndarray): 传感器节点索引列表
    enhancement_factor (float): 增强因子

    Returns:
    numpy.ndarray: 增强后的邻接矩阵
    """
    # 创建一个与邻接矩阵同形状的mask，初始化为1
    sensor_indices = np.where(sensor_mask == 1)[0]
    mask = np.ones_like(adj_matrix)

    # 设置传感器节点的行和列的mask值
    mask[sensor_indices, :] *= enhancement_factor
    mask[:, sensor_indices] *= enhancement_factor
    # 应用mask到原邻接矩阵
    enhanced_adj_matrix = adj_matrix * mask
    return enhanced_adj_matrix

def add_self_connections_to_sensors(adj_matrix, sensor_mask, self_weight=1.0):
    """
    在邻接矩阵中为传感器节点添加自连接权重。
    
    参数:
    adj_matrix (np.array): 原始的加权邻接矩阵。
    sensor_mask (np.array): 一个一维数组，其中0表示没有传感器，1表示有传感器。
    self_weight (float): 要添加到自连接的权重。

    返回:
    np.array: 修改后的邻接矩阵。
    """
    # 找到传感器位置的索引
    sensor_indices = np.where(sensor_mask == 1)[0]
    # 在对角线位置为传感器节点增加权重
    adj_matrix[sensor_indices, sensor_indices] += self_weight
    return adj_matrix

def create_graph_from_adjacency_matrix(adj_matrix):
    """从邻接矩阵创建有向图，并添加边和权重"""
    num_nodes = adj_matrix.shape[0]
    print(num_nodes)
    G = nx.DiGraph()
    for i in range(num_nodes):
        for j in range(num_nodes):
            if adj_matrix[i, j] != 0 :  # 假设0表示没有直接连接
                G.add_edge(i, j, weight=adj_matrix[i, j])
    return G
def compute_shortest_paths(G, sensor_indices):
    """这是用来计算有向图中传感器节点到各个节点的最短距离的"""
    shortest_paths = {}
    for sensor_idx in sensor_indices:
        # 计算从每个传感器节点出发到所有其他节点的最短路径和长度
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
    
    weights = 1 / (min_distances + 1)  # 防止除零错误
    weights[min_distances == float('inf')] = 0
    weight_sums = weights.sum(axis=0, keepdim=True)  # Sum across sensor nodes for each non-sensor node
    normalized_weights = weights / weight_sums  # Normalize weights
    # 检查weights是否包含nan值
    if torch.isnan(normalized_weights).any():
        print("Warning: weights contains nan values.")
        normalized_weights = torch.nan_to_num(normalized_weights, nan=0.0) 
    
    return normalized_weights.transpose(0, 1)  # 转置以使维度正确

def initialize_features_with_min_paths(attr_matrix, shortest_paths_fwd, shortest_paths_rev, sensor_indices, non_sensor_indices):
    # 检查attr_matrix的维度并扩展最后一个维度
    if attr_matrix.dim() == 2:
        attr_matrix = attr_matrix.unsqueeze(-1)     
    weights = compute_weight_matrix({'fwd': shortest_paths_fwd, 'rev': shortest_paths_rev}, sensor_indices, non_sensor_indices)
    weights_expanded = weights.unsqueeze(0).expand(attr_matrix.size(0), -1, -1)
    
    # 获取传感器特征
    sensor_features = attr_matrix[:, sensor_indices, :]  # 维度 [batch_size, num_sensors, feature_size]
    
    # 计算加权特征
    weighted_features = torch.bmm(weights_expanded, sensor_features)  # [batch_size, num_non_sensors, feature_size]    
    # 更新非传感器节点的特征
    attr_matrix[:, non_sensor_indices, :] = weighted_features
    if torch.isnan(attr_matrix).any():
        print("Warning: weights contains nan values.")
    if torch.isinf(attr_matrix).any():
        print("Warning: weights contains inf values.")
    return attr_matrix


def train_model(model, batch_size,feature_size,output_size, epochs,steps_per_epoch,TrainDataset, TestDataset,adj_matrices, dist_matrices,mask_0,mask_1,mask_2,mask_area=None,yanmo=True,percent=0.2,loss_type=0,standard_type=0,threshold=0.0001,init=False,w=1):
    #w是超参数，用来平衡物理损失和预测精度的权重，默认是1
    
       
    NODE_NUM=adj_matrices.shape[0]
    FEATURE_NUM=feature_size
    STEPS_PER_EPOCH = steps_per_epoch

    EPOCHS = epochs  
    BATCH_SIZE = batch_size
    VAL_BATCH_SIZE = min(len(TestDataset),batch_size)

    
    #这个mask作为输入的mask(非传感器节点的序号)
    mask_measure1=np.argwhere(mask_0==0)[:,0]

    mask_sample=mask_1 
    
    # 对每个区域的节点属性进行标准化，并保存为一个矩阵
    # 创建MinMaxScaler对象
    region_params = {}
    TrainDataset= torch.tensor(TrainDataset, dtype=torch.float32)
    TestDataset= torch.tensor(TestDataset, dtype=torch.float32)
    
    #在这主要是为了给传感器添加一个自连接节点
    self_attention=False
    if self_attention==True:
        # 调用函数添加自连接
        adj_matrices = add_self_connections_to_sensors(adj_matrices, mask_0, self_weight=1.0)
    
    if standard_type==0:
        pass
    elif standard_type == 1:        
        # 计算所有区域的均值和标准差
        mean_value = torch.mean(TrainDataset)
        std_value = torch.std(TrainDataset)

        # 对训练集和测试集进行z-score标准化
        TrainDataset = (TrainDataset - mean_value) / std_value
        TestDataset = (TestDataset - mean_value) / std_value
        # 保存均值和标准差
        region_params = {'mean': mean_value, 'std': std_value}
    elif standard_type == 2:
        # 分不同区域进行最大最小标准化
        for region in range(1, 5):  # 区域编号从1到4
            region_indices = torch.nonzero(mask_area == region).squeeze()  # 获取属于当前区域的节点索引            
            mean_value = torch.mean(TrainDataset[:, :, region_indices])
            std_value = torch.std(TrainDataset[:, :, region_indices])            
            # 对训练集和测试集进行z-score标准化
            TrainDataset[:, :, region_indices] = (TrainDataset[:, :, region_indices] - mean_value) / std_value
            TestDataset[:, :, region_indices] = (TestDataset[:, :, region_indices] - mean_value) / std_value

            region_params[region] = {'mean': mean_value, 'std': std_value}
            
    # Split train dataset in batches
    train_data = DataLoader(TrainDataset, batch_size=BATCH_SIZE, shuffle=True)
    val_data = DataLoader(TestDataset, batch_size=VAL_BATCH_SIZE, shuffle=False) 
        
    #criterion = nn.NLLLoss()
    # #做回归的时候用的是MSE的损失函数
    # criterion = nn.MSELoss()
    #做回归的时候用的是MAE的损失函数
    # criterion = nn.L1Loss()  

    optimizer = torch.optim.Adam(model.parameters())        
    # Initial evaluations
    # =========================================================================    
    loss_ls = []
    mae_ls,mse_ls,rmse_ls,mape_ls,mspe_ls = [],[],[],[],[]


    
    #adj_matrices=enhance_sensor_impact_np(adj_matrices, mask_0, enhancement_factor=10)
    
    
    G = create_graph_from_adjacency_matrix(dist_matrices)
    sensor_indices=np.argwhere(mask_0==1)[:,0]        
    # 计算正向最短路径
    shortest_paths_fwd = compute_shortest_paths(G, sensor_indices)

    # 计算反向最短路径
    G_rev = G.reverse()
    shortest_paths_rev = compute_shortest_paths(G_rev, sensor_indices)
    a = adj_matrices
    # d = dist_matrices
    A_base = torch.from_numpy(a).float().cuda()  # 基础 A 矩阵
    # D_base = torch.from_numpy(d).float().cuda()  # 基础 D 矩阵

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
                    #这里需要复制mask_0,因为后面的test得用所有的数据来进行输入，如果这改写了会影响后面的test结果
                    #如果掩膜为真，那么就在训练的时候随机选择20%的传感器的数字设置为0，然后进行训练
                    mask_00=mask_0.copy()
                    
                    #indices=np.where(mask_00 == 1)[0]
                    #现在更新一下掩膜的站点：用mask_sample（只是说在mask_sample里面传感器的节点里面选掩膜的节点，但是还是应该在mask_0里面进行掩膜）
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
                #     #需要注意到如果进行掩膜操作的话每一次的掩膜都会变，所以后面的部分测试集掩膜要保证跟刚开始一样
                #     mask_measure_1= np.argwhere(mask_0==0)[:,0]  
                #     X[:,mask_measure_1,:]=0


                
 
                                         
                # 调整 A 和 D 的形状以匹配批次大小
                A = A_base.expand(X.shape[0], -1, -1)  # 使用 expand 节省内存
                Y = model(X.cuda(),A)
                
                if standard_type==0:
                    pass
                elif standard_type == 1:
                    # # 所有区域统一进行最大最小标准化
                    # Y = Y * (region_params['max']-region_params['min']) + min_value
                    # T = T * (region_params['max']-region_params['min']) + min_value
                    
                    # 所有区域统一进行z-score标准化
                    mean_value = region_params['mean']
                    std_value = region_params['std']
                    Y = Y*std_value + mean_value
                    T = T*std_value + mean_value

                elif standard_type == 2:
                    # 分不同区域进行最大最小标准化
                    for region in range(1, 5):  # 区域编号从1到4
                        region_indices = torch.nonzero(mask_area == region).squeeze()  # 获取属于当前区域的节点索引
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
            
            #添加初始的特征属性
            if percent==0 and init:
                sensor_indices=np.argwhere(mask_0==1)[:,0]
                non_sensor_indices=np.argwhere(mask_0==0)[:,0]                
                X = initialize_features_with_min_paths(X, shortest_paths_fwd, shortest_paths_rev, sensor_indices, non_sensor_indices)
            
            T=batch[:,:,-output_size:]
                           
            # 调整 A 和 D 的形状以匹配批次大小
            A = A_base.expand(X.shape[0], -1, -1)  # 使用 expand 节省内存
            Y = model(X.cuda(),A) 
            
            if standard_type==0:
                pass
            elif standard_type == 1:
                
                # 所有区域统一进行z-score标准化
                mean_value = region_params['mean']
                std_value = region_params['std']
                Y = Y*std_value + mean_value
                T = T*std_value + mean_value

            elif standard_type == 2:
                # 分不同区域进行最大最小标准化
                for region in range(1, 5):  # 区域编号从1到4
                    region_indices = torch.nonzero(mask_area == region).squeeze()  # 获取属于当前区域的节点索引                    
                    mean_value = region_params[region]['mean']
                    std_value = region_params[region]['std']
                    Y[:, region_indices] = Y[:, region_indices]*std_value + mean_value
                    T[:, region_indices] = T[:, region_indices]*std_value + mean_value   
                        
            Y__=Y.detach().cpu().numpy()
            T__=T.detach().cpu().numpy()
                        
            nonzero_cols1 = np.argwhere(mask_2==1)[:,0]
            #选择不全为0的列构成新的张量,而且只取最后一个值进行比较
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
