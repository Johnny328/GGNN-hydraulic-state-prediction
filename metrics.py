import numpy as np
import torch
import torch.nn as nn

def RSE(pred, true):
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))

def CORR(pred, true):
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    return (u / d).mean(-1)

def MAE(pred, true):
    return np.mean(np.abs(pred - true))

def MSE(pred, true):
    return np.mean((pred - true) ** 2)

def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))

def MAPE(pred, true):
    return np.mean(np.abs((pred - true) / (true+1e-5)))

def MSPE(pred, true):
    return np.mean(np.square((pred - true) / (true+1e-5)))

# def restrict(Y, pred_bounds):
#     # 确保 pred_bounds 转换为与 Y 相同的数据类型
#     pred_bounds = torch.tensor(pred_bounds, dtype=Y.dtype).cuda()
    
#     # 获取下界和上界
#     lower_bounds, upper_bounds = pred_bounds[:, 0], pred_bounds[:, 1]
    
#     # 扩展下界和上界以匹配 Y 的批次大小
#     lower_bounds = lower_bounds.unsqueeze(0).expand_as(Y)
#     upper_bounds = upper_bounds.unsqueeze(0).expand_as(Y)
    
#     # 将 Y 中超出界限的值约束为上界或下界
#     Y_clipped = torch.min(torch.max(Y, lower_bounds), upper_bounds)
    
#     return Y_clipped

def restrict(Y, pred_bounds,mask_measure):
    # 将 Y 和 pred_bounds 转换为 PyTorch 张量并移动到 GPU
    Y=Y.cuda()
    # 确保 pred_bounds 转换为与 Y 相同的数据类型
    pred_bounds = torch.tensor(pred_bounds, dtype=Y.dtype).cuda()
    mask_measure = torch.tensor(mask_measure, dtype=torch.bool).cuda()    
    # 获取下界和上界
    lower_bounds, upper_bounds = pred_bounds[:, 0], pred_bounds[:, 1]
    
    # 扩展下界和上界以匹配 Y 的批次大小
    lower_bounds = lower_bounds.unsqueeze(0).expand_as(Y)
    upper_bounds = upper_bounds.unsqueeze(0).expand_as(Y)   
     
    # 使用 mask_measure 来保护传感器站点的原始预测值
    # 只对非传感器站点的节点应用约束
    mask_no_measure = ~mask_measure  # 获取非传感器站点的掩膜
    mask_no_measure = mask_no_measure.unsqueeze(0).expand_as(Y)
        
    # 创建一个复制 Y 的张量来进行条件修改
    Y_clipped = Y.clone()

    # 将 Y 中超出界限的非传感器站点值约束为上界或下界
    Y_clipped[mask_no_measure] = torch.min(torch.max(Y[mask_no_measure], lower_bounds[mask_no_measure]), upper_bounds[mask_no_measure])

    return Y_clipped
    

#计算常规的损失（考虑检测站点的损失以及非监测站点的损失
def custom_loss(Y, T, pred_bounds, mask_measure,mask_prediction):
    
    Y=Y.cuda()
    T=T.cuda()
    pred_bounds = torch.tensor(pred_bounds, dtype=torch.float32).cuda()  # 确保类型匹配
    mask_measure = torch.tensor(mask_measure, dtype=torch.bool).cuda()  # 使用bool类型更合适
    mask_prediction = torch.tensor(mask_prediction, dtype=torch.bool).cuda()  # 使用bool类型更合适

    # 获取有监测数据的节点索引
    mask_measure_idx = torch.nonzero(mask_measure, as_tuple=True)[0]

    # 获取无监测数据的节点索引
    #mask_no_measure_idx = torch.nonzero(~mask_measure, as_tuple=True)[0]
    #我们在比较的时候不用比较没有节点需水量的点，这些点忽略不计(又要是有需水量的，又要是非传感器的)
    mask_intersection = torch.logical_and(~mask_measure, mask_prediction)
    mask_no_measure_idx=torch.nonzero(mask_intersection, as_tuple=True)[0]
    
    
    #mask_prediction_idx = torch.nonzero(mask_prediction, as_tuple=True)[0]


    # 计算基本损失（MAE） - 只考虑有监测数据的节点(有数据的部分取平均)
    #mae_loss=MAE(Y[:, mask_measure_idx],T[:, mask_measure_idx])
    mae_loss = torch.mean(torch.abs(Y[:, mask_measure_idx] - T[:, mask_measure_idx]))
    # criterion = nn.L1Loss()
    # mae_loss = criterion(Y[:, mask_measure_idx] , T[:, mask_measure_idx])

    # 计算额外损失 - 只考虑无监测数据的节点
    extra_loss = 0
    if len(mask_no_measure_idx) > 0:
        pred_no_measure = Y[:, mask_no_measure_idx]
        lower_bounds = pred_bounds[mask_no_measure_idx, 0].unsqueeze(0)
        upper_bounds = pred_bounds[mask_no_measure_idx, 1].unsqueeze(0)

        # 计算超出上下界的损失
        lower_losses = torch.clamp(lower_bounds - pred_no_measure, min=0)
        upper_losses = torch.clamp(pred_no_measure - upper_bounds, min=0)

        # 将超出上下界的损失相加
        #extra_loss = torch.mean(lower_losses + upper_losses)
        
    # 计算总损失
    total_loss = mae_loss + 0.1*extra_loss
    #total_loss = mae_loss 
    return total_loss

import torch

def calculate_physical_constraints_loss(Y, adj_matrix, threshold=0.0001):
    # 将邻接矩阵转换为0-1矩阵，大于0的设为1
    binary_adj_matrix = (adj_matrix > 0).float()

    if Y.dim() == 2:  # 处理二维情况
        # 扩展Y以便进行逐元素操作
        Y_expanded = Y.unsqueeze(-1).expand(-1, -1, Y.size(1))

        # 计算节点间的水头差，上游节点的水头需要大于下游节点
        head_diff = Y_expanded - Y_expanded.transpose(1, 2)

        # 应用邻接矩阵过滤，只考虑实际连接的节点间差异
        constrained_head_diff = head_diff * binary_adj_matrix

        # 只对违反物理约束（下游水头+阈值大于上游水头）的情况计算损失
        violations = torch.relu(-(constrained_head_diff - threshold))

        # 计算损失，求和并平均
        total_violations = torch.sum(violations)
        num_connections = torch.sum(binary_adj_matrix)
        mean_violation = total_violations / num_connections if num_connections > 0 else torch.tensor(0.0).to(Y.device)
    elif Y.dim() == 3:  # 处理三维情况
        # 扩展Y以便进行逐元素操作
        Y_expanded = Y.unsqueeze(2).expand(-1, -1, Y.size(1), -1)

        # 计算节点间的水头差（针对每个时间步）
        head_diff = Y_expanded - Y_expanded.transpose(1, 2)

        # 应用邻接矩阵过滤，只考虑实际连接的节点间差异
        constrained_head_diff = head_diff * binary_adj_matrix.unsqueeze(-1)

        # 只对违反物理约束（下游水头+阈值大于上游水头）的情况计算损失
        violations = torch.relu(-(constrained_head_diff - threshold))

        # 计算损失，沿节点和时间步求和并平均
        total_violations = torch.sum(violations)
        num_connections = torch.sum(binary_adj_matrix) * Y.size(-1)  # 考虑时间维度的连接总数
        mean_violation = total_violations / num_connections if num_connections > 0 else torch.tensor(0.0).to(Y.device)
    else:
        raise ValueError("Y should be either 2D or 3D")

    return mean_violation
    

#计算常规的损失（考虑检测站点的损失以及非监测站点的损失
def custom_loss_extend(Y, T, mask_measure,mask_prediction):
    
    Y=Y.cuda()
    T=T.cuda()
    mask_measure = torch.tensor(mask_measure, dtype=torch.bool).cuda()  # 使用bool类型更合适
    mask_prediction = torch.tensor(mask_prediction, dtype=torch.bool).cuda()  # 使用bool类型更合适
    # 获取有监测数据的节点索引
    mask_measure_idx = torch.nonzero(mask_measure, as_tuple=True)[0]
    # 获取无监测数据的节点索引
    mask_no_measure_idx = torch.nonzero(~mask_measure, as_tuple=True)[0]
    #我们在比较的时候不用比较没有节点需水量的点，这些点忽略不计(又要是有需水量的，又要是非传感器的)
    mask_intersection = torch.logical_and(~mask_measure, mask_prediction)
    mask_no_measure_idx=torch.nonzero(mask_intersection, as_tuple=True)[0]
    

    # 计算基本损失（MAE） - 只考虑有监测数据的节点(有数据的部分取平均)
    mae_loss_1 = torch.mean(torch.abs(Y[:, mask_measure_idx] - T[:, mask_measure_idx]))
    # 计算额外损失，用算法估计未监测点的值，然后用它作为参考来更新 - 只考虑有监测数据的节点(有数据的部分取平均)
    mae_loss_2 = torch.mean(torch.abs(Y[:, mask_no_measure_idx] - T[:, mask_no_measure_idx]))
        
    # 计算总损失
    total_loss = 0.6*mae_loss_1 + 0.4*mae_loss_2
    #total_loss = mae_loss 
    return total_loss


#计算常规的损失（考虑检测站点的损失以及物理约束
def custom_loss_extend_physic1(Y, T, A,mask_measure,mask_prediction):
    
    Y=Y.cuda()
    T=T.cuda()
    A=A.cuda()
    mask_measure = torch.tensor(mask_measure, dtype=torch.bool).cuda()  # 使用bool类型更合适
    mask_prediction = torch.tensor(mask_prediction, dtype=torch.bool).cuda()  # 使用bool类型更合适
    # 获取有监测数据的节点索引
    mask_measure_idx = torch.nonzero(mask_measure, as_tuple=True)[0]
    # 获取无监测数据的节点索引
    mask_no_measure_idx = torch.nonzero(~mask_measure, as_tuple=True)[0]
    #我们在比较的时候不用比较没有节点需水量的点，这些点忽略不计(又要是有需水量的，又要是非传感器的)
    mask_intersection = torch.logical_and(~mask_measure, mask_prediction)
    mask_no_measure_idx=torch.nonzero(mask_intersection, as_tuple=True)[0]
    

    # 计算基本损失（MAE） - 只考虑有监测数据的节点(有数据的部分取平均)
    mae_loss_1 = torch.mean(torch.abs(Y[:, mask_measure_idx] - T[:, mask_measure_idx]))
    #计算物理约束的损失
    mae_loss_2 =calculate_physical_constraints_loss(Y, A)
    # 计算总损失
    total_loss = mae_loss_1 +mae_loss_2
    #total_loss = mae_loss 
    return total_loss

#计算常规的损失（考虑检测站点的损失以及非监测站点的损失以及物理约束(这个是最终论文设置的情况)
def custom_loss_extend_physic(Y, T, A,mask_measure,mask_prediction,loss_type=0,threshold=0.0001,w=1):
    
    Y=Y.cuda()
    T=T.cuda()
    A=A.cuda()
    mask_measure = torch.tensor(mask_measure, dtype=torch.bool).cuda()  # 使用bool类型更合适
    mask_prediction = torch.tensor(mask_prediction, dtype=torch.bool).cuda()  # 使用bool类型更合适
    # 获取有监测数据的节点索引
    mask_measure_idx = torch.nonzero(mask_measure, as_tuple=True)[0]
    # 获取无监测数据的节点索引
    mask_no_measure_idx = torch.nonzero(~mask_measure, as_tuple=True)[0]
    #我们在比较的时候不用比较没有节点需水量的点，这些点忽略不计(又要是有需水量的，又要是非传感器的)
    mask_intersection = torch.logical_and(~mask_measure, mask_prediction)
    mask_no_measure_idx=torch.nonzero(mask_intersection, as_tuple=True)[0]
    
    if loss_type==0:
        # 计算基本损失（MAE） - 只考虑有监测数据的节点(有数据的部分取平均)
        total_loss = torch.mean(torch.abs(Y[:, mask_measure_idx] - T[:, mask_measure_idx]))
        
    elif loss_type==1:        
        # 计算基本损失（MAE） - 只考虑有监测数据的节点(有数据的部分取平均)
        mae_loss_1 = torch.mean(torch.abs(Y[:, mask_measure_idx] - T[:, mask_measure_idx]))
        #计算物理约束的损失
        mae_loss_2 =calculate_physical_constraints_loss(Y, A,threshold)
        # 计算总损失
        total_loss = mae_loss_1 +mae_loss_2*w        
    elif loss_type==2:        
        # 计算基本损失（MAE） - 只考虑有监测数据的节点(有数据的部分取平均)
        mae_loss_1 = torch.mean(torch.abs(Y[:, mask_measure_idx] - T[:, mask_measure_idx]))
        # 计算额外损失，用算法估计未监测点的值，然后用它作为参考来更新 - 只考虑有监测数据的节点(有数据的部分取平均)
        mae_loss_2 = torch.mean(torch.abs(Y[:, mask_no_measure_idx] - T[:, mask_no_measure_idx]))     
        total_loss = 0.7*mae_loss_1 + 0.3*mae_loss_2
    else:
        # 如果 loss_type 不在 [0, 1, 2] 中，抛出一个异常
        raise ValueError(f"Invalid loss_type: {loss_type}. Expected one of 0, 1, 2.")        
    return total_loss

def metric(pred, true):
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)

    return mae,mse,rmse,mape,mspe


# def metric(pred, real):
#     mae = masked_mae_np(pred, real, 0.0).item()
#     mse = masked_mse_np(pred, real, 0.0).item()
#     rmse = masked_rmse_np(pred, real, 0.0).item()
#     mape = masked_mape_np(pred, real, 0.0).item()
#     mspe = masked_mspe_np(pred, real, 0.0).item()

#     return mae, mse, rmse, mape, mspe

## the loss metric for tensors
def masked_mse(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    elif null_val == 0:
        mask = (labels > 0).float() ## the value maybe < 0
    else:
        mask = (labels != null_val)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = (preds - labels) ** 2
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def masked_rmse(preds, labels, null_val=np.nan):
    return torch.sqrt(masked_mse(preds=preds, labels=labels, null_val=null_val))


def masked_mae(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    elif null_val == 0:
        mask = (labels > 0).float()
    else:
        mask = (labels != null_val)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds - labels)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)

def masked_mape(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    elif null_val == 0:
        mask = (labels > 0).float()
    else:
        mask = (labels != null_val)

    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds - labels) / labels
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)

## the metric computation over array data
def masked_rmse_np(preds, labels, null_val=np.nan):
    return np.sqrt(masked_mse_np(preds=preds, labels=labels, null_val=null_val))


def masked_mse_np(preds, labels, null_val=np.nan):
    with np.errstate(divide='ignore', invalid='ignore'):
        if np.isnan(null_val):
            mask = ~np.isnan(labels)
        elif null_val == 0:
            mask = (labels > 0).astype(np.float32)
        else:
            mask = np.not_equal(labels, null_val)
        mask = mask.astype('float32')
        mask /= np.mean(mask)
        rmse = np.square(np.subtract(preds, labels)).astype('float32')
        rmse = np.nan_to_num(rmse * mask)
        return np.mean(rmse)


def masked_mae_np(preds, labels, null_val=np.nan):
    with np.errstate(divide='ignore', invalid='ignore'):
        if np.isnan(null_val):
            mask = ~np.isnan(labels)
        elif null_val == 0:
            mask = (labels > 0).astype(np.float32)
        else:
            mask = np.not_equal(labels, null_val)
        mask = mask.astype('float32')
        mask /= np.mean(mask)
        mae = np.abs(np.subtract(preds, labels)).astype('float32')
        mae = np.nan_to_num(mae * mask)
        return np.mean(mae)


def masked_mape_np(preds, labels, null_val=np.nan):
    with np.errstate(divide='ignore', invalid='ignore'):
        if np.isnan(null_val):
            mask = ~np.isnan(labels)
        elif null_val == 0:
            mask = (labels > 0).astype(np.float32)
        else:
            mask = np.not_equal(labels, null_val)
        mask = mask.astype('float32')
        mask /= np.mean(mask)
        mape = np.abs(np.divide(np.subtract(preds, labels).astype('float32'), labels))
        mape = np.nan_to_num(mask * mape)
        return np.mean(mape)


def masked_mspe_np(preds, labels, null_val=np.nan):
    with np.errstate(divide='ignore', invalid='ignore'):
        if np.isnan(null_val):
            mask = ~np.isnan(labels)
        elif null_val == 0:
            mask = (labels > 0).astype(np.float32)
        else:
            mask = np.not_equal(labels, null_val)
        mask = mask.astype('float32')
        mask /= np.mean(mask)

        mspe = np.square(np.divide(np.subtract(preds, labels).astype('float32'), labels))
        mspe = np.nan_to_num(mask * mspe)
        return np.mean(mspe)