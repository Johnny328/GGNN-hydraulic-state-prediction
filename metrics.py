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
#     # Ensure pred_bounds is converted to the same data type as Y
#     pred_bounds = torch.tensor(pred_bounds, dtype=Y.dtype).cuda()
    
#     # Get lower and upper bounds
#     lower_bounds, upper_bounds = pred_bounds[:, 0], pred_bounds[:, 1]
    
#     # Expand lower and upper bounds to match Y's batch size
#     lower_bounds = lower_bounds.unsqueeze(0).expand_as(Y)
#     upper_bounds = upper_bounds.unsqueeze(0).expand_as(Y)
    
#     # Clip Y to the bounds
#     Y_clipped = torch.min(torch.max(Y, lower_bounds), upper_bounds)
    
#     return Y_clipped

def restrict(Y, pred_bounds,mask_measure):
    # Convert Y and pred_bounds to PyTorch tensors and move to GPU
    Y=Y.cuda()
    # Ensure pred_bounds is converted to the same data type as Y
    pred_bounds = torch.tensor(pred_bounds, dtype=Y.dtype).cuda()
    mask_measure = torch.tensor(mask_measure, dtype=torch.bool).cuda()    
    # Get lower and upper bounds
    lower_bounds, upper_bounds = pred_bounds[:, 0], pred_bounds[:, 1]
    
    # Expand lower and upper bounds to match Y's batch size
    lower_bounds = lower_bounds.unsqueeze(0).expand_as(Y)
    upper_bounds = upper_bounds.unsqueeze(0).expand_as(Y)   
     
    # Use mask_measure to protect original predicted values of sensor sites
    # Only apply constraints to non-sensor site nodes
    mask_no_measure = ~mask_measure  # Get mask for non-sensor sites
    mask_no_measure = mask_no_measure.unsqueeze(0).expand_as(Y)
        
    # Create a copy of Y tensor for conditional modification
    Y_clipped = Y.clone()

    # Constrain non-sensor site values in Y that exceed bounds to upper or lower bounds
    Y_clipped[mask_no_measure] = torch.min(torch.max(Y[mask_no_measure], lower_bounds[mask_no_measure]), upper_bounds[mask_no_measure])

    return Y_clipped
    

# Calculate the conventional loss (considering the loss of detection sites and non-monitored sites)
def custom_loss(Y, T, pred_bounds, mask_measure,mask_prediction):
    
    Y=Y.cuda()
    T=T.cuda()
    pred_bounds = torch.tensor(pred_bounds, dtype=torch.float32).cuda()  # Ensure type matching
    mask_measure = torch.tensor(mask_measure, dtype=torch.bool).cuda()  # Using bool type is more appropriate
    mask_prediction = torch.tensor(mask_prediction, dtype=torch.bool).cuda()  # Using bool type is more appropriate

    # Get indices of nodes with monitored data
    mask_measure_idx = torch.nonzero(mask_measure, as_tuple=True)[0]

    # Get indices of nodes without monitored data
    # When comparing, we don't compare points without node water demand, these points are ignored (must have water demand and be non-sensors)
    mask_intersection = torch.logical_and(~mask_measure, mask_prediction)
    mask_no_measure_idx=torch.nonzero(mask_intersection, as_tuple=True)[0]
    
    
    #mask_prediction_idx = torch.nonzero(mask_prediction, as_tuple=True)[0]


    # Calculate basic loss (MAE) - only consider nodes with monitored data (average the parts with data)
    #mae_loss=MAE(Y[:, mask_measure_idx],T[:, mask_measure_idx])
    mae_loss = torch.mean(torch.abs(Y[:, mask_measure_idx] - T[:, mask_measure_idx]))
    # criterion = nn.L1Loss()
    # mae_loss = criterion(Y[:, mask_measure_idx] , T[:, mask_measure_idx])

    # Calculate additional loss - only consider nodes without monitored data
    extra_loss = 0
    if len(mask_no_measure_idx) > 0:
        pred_no_measure = Y[:, mask_no_measure_idx]
        lower_bounds = pred_bounds[mask_no_measure_idx, 0].unsqueeze(0)
        upper_bounds = pred_bounds[mask_no_measure_idx, 1].unsqueeze(0)

        # Calculate losses exceeding upper and lower bounds
        lower_losses = torch.clamp(lower_bounds - pred_no_measure, min=0)
        upper_losses = torch.clamp(pred_no_measure - upper_bounds, min=0)

        # Add the losses exceeding upper and lower bounds
        #extra_loss = torch.mean(lower_losses + upper_losses)
        
    # Calculate total loss
    total_loss = mae_loss + 0.1*extra_loss
    #total_loss = mae_loss 
    return total_loss

import torch

def calculate_physical_constraints_loss(Y, adj_matrix, threshold=0.0001):
    # Convert adjacency matrix to 0-1 matrix, set values greater than 0 to 1
    binary_adj_matrix = (adj_matrix > 0).float()

    if Y.dim() == 2:  # Handle 2D case
        # Expand Y for element-wise operations
        Y_expanded = Y.unsqueeze(-1).expand(-1, -1, Y.size(1))

        # Calculate head difference between nodes, upstream node head should be greater than downstream
        head_diff = Y_expanded - Y_expanded.transpose(1, 2)

        # Apply adjacency matrix filter, only consider differences between actually connected nodes
        constrained_head_diff = head_diff * binary_adj_matrix

        # Only calculate loss for cases violating physical constraints (downstream head + threshold > upstream head)
        violations = torch.relu(-(constrained_head_diff - threshold))

        # Calculate loss, sum and average
        total_violations = torch.sum(violations)
        num_connections = torch.sum(binary_adj_matrix)
        mean_violation = total_violations / num_connections if num_connections > 0 else torch.tensor(0.0).to(Y.device)
    elif Y.dim() == 3:  # Handle 3D case
        # Expand Y for element-wise operations
        Y_expanded = Y.unsqueeze(2).expand(-1, -1, Y.size(1), -1)

        # Calculate head difference between nodes (for each time step)
        head_diff = Y_expanded - Y_expanded.transpose(1, 2)

        # Apply adjacency matrix filter, only consider differences between actually connected nodes
        constrained_head_diff = head_diff * binary_adj_matrix.unsqueeze(-1)

        # Only calculate loss for cases violating physical constraints (downstream head + threshold > upstream head)
        violations = torch.relu(-(constrained_head_diff - threshold))

        # Calculate loss, sum and average along nodes and time steps
        total_violations = torch.sum(violations)
        num_connections = torch.sum(binary_adj_matrix) * Y.size(-1)  # Consider total connections in time dimension
        mean_violation = total_violations / num_connections if num_connections > 0 else torch.tensor(0.0).to(Y.device)
    else:
        raise ValueError("Y should be either 2D or 3D")

    return mean_violation
    

# Calculate the conventional loss (considering the loss of detection sites and non-monitored sites
def custom_loss_extend(Y, T, mask_measure,mask_prediction):
    
    Y=Y.cuda()
    T=T.cuda()
    mask_measure = torch.tensor(mask_measure, dtype=torch.bool).cuda()  # Using bool type is more appropriate
    mask_prediction = torch.tensor(mask_prediction, dtype=torch.bool).cuda()  # Using bool type is more appropriate
    # Get indices of nodes with monitored data
    mask_measure_idx = torch.nonzero(mask_measure, as_tuple=True)[0]
    # Get indices of nodes without monitored data
    mask_no_measure_idx = torch.nonzero(~mask_measure, as_tuple=True)[0]
    # When comparing, we don't compare points without node water demand, these points are ignored (must have water demand and be non-sensors)
    mask_intersection = torch.logical_and(~mask_measure, mask_prediction)
    mask_no_measure_idx=torch.nonzero(mask_intersection, as_tuple=True)[0]
    

    # Calculate basic loss (MAE) - only consider nodes with monitored data (average the parts with data)
    mae_loss_1 = torch.mean(torch.abs(Y[:, mask_measure_idx] - T[:, mask_measure_idx]))
    # Calculate additional loss, use algorithm to estimate the value of unmonitored points, then use it as reference to update - only consider nodes with monitored data (average the parts with data)
    mae_loss_2 = torch.mean(torch.abs(Y[:, mask_no_measure_idx] - T[:, mask_no_measure_idx]))
        
    # Calculate total loss
    total_loss = 0.6*mae_loss_1 + 0.4*mae_loss_2
    #total_loss = mae_loss 
    return total_loss


# Calculate the conventional loss (considering the loss of detection sites and physical constraints)
def custom_loss_extend_physic1(Y, T, A,mask_measure,mask_prediction):
    
    Y=Y.cuda()
    T=T.cuda()
    A=A.cuda()
    mask_measure = torch.tensor(mask_measure, dtype=torch.bool).cuda()  # Using bool type is more appropriate
    mask_prediction = torch.tensor(mask_prediction, dtype=torch.bool).cuda()  # Using bool type is more appropriate
    # Get indices of nodes with monitored data
    mask_measure_idx = torch.nonzero(mask_measure, as_tuple=True)[0]
    # Get indices of nodes without monitored data
    mask_no_measure_idx = torch.nonzero(~mask_measure, as_tuple=True)[0]
    # When comparing, we don't compare points without node water demand, these points are ignored (must have water demand and be non-sensors)
    mask_intersection = torch.logical_and(~mask_measure, mask_prediction)
    mask_no_measure_idx=torch.nonzero(mask_intersection, as_tuple=True)[0]
    

    # Calculate basic loss (MAE) - only consider nodes with monitored data (average the parts with data)
    mae_loss_1 = torch.mean(torch.abs(Y[:, mask_measure_idx] - T[:, mask_measure_idx]))
    # Calculate the loss of physical constraints
    mae_loss_2 =calculate_physical_constraints_loss(Y, A)
    # Calculate total loss
    total_loss = mae_loss_1 +mae_loss_2
    #total_loss = mae_loss 
    return total_loss

# Calculate the conventional loss (considering the loss of detection sites, non-monitored sites, and physical constraints (this is the final paper setting))
def custom_loss_extend_physic(Y, T, A,mask_measure,mask_prediction,loss_type=0,threshold=0.0001,w=1):
    
    Y=Y.cuda()
    T=T.cuda()
    A=A.cuda()
    mask_measure = torch.tensor(mask_measure, dtype=torch.bool).cuda()  # Using bool type is more appropriate
    mask_prediction = torch.tensor(mask_prediction, dtype=torch.bool).cuda()  # Using bool type is more appropriate
    # Get indices of nodes with monitored data
    mask_measure_idx = torch.nonzero(mask_measure, as_tuple=True)[0]
    # Get indices of nodes without monitored data
    mask_no_measure_idx = torch.nonzero(~mask_measure, as_tuple=True)[0]
    # When comparing, we don't compare points without node water demand, these points are ignored (must have water demand and be non-sensors)
    mask_intersection = torch.logical_and(~mask_measure, mask_prediction)
    mask_no_measure_idx=torch.nonzero(mask_intersection, as_tuple=True)[0]
    
    if loss_type==0:
        # Calculate basic loss (MAE) - only consider nodes with monitored data (average the parts with data)
        total_loss = torch.mean(torch.abs(Y[:, mask_measure_idx] - T[:, mask_measure_idx]))
        
    elif loss_type==1:        
        # Calculate basic loss (MAE) - only consider nodes with monitored data (average the parts with data)
        mae_loss_1 = torch.mean(torch.abs(Y[:, mask_measure_idx] - T[:, mask_measure_idx]))
        # Calculate the loss of physical constraints
        mae_loss_2 =calculate_physical_constraints_loss(Y, A,threshold)
        # Calculate total loss
        total_loss = mae_loss_1 +mae_loss_2*w        
    elif loss_type==2:        
        # Calculate basic loss (MAE) - only consider nodes with monitored data (average the parts with data)
        mae_loss_1 = torch.mean(torch.abs(Y[:, mask_measure_idx] - T[:, mask_measure_idx]))
        # Calculate additional loss, use algorithm to estimate the value of unmonitored points, then use it as reference to update - only consider nodes with monitored data (average the parts with data)
        mae_loss_2 = torch.mean(torch.abs(Y[:, mask_no_measure_idx] - T[:, mask_no_measure_idx]))     
        total_loss = 0.7*mae_loss_1 + 0.3*mae_loss_2
    else:
        # If loss_type is not in [0, 1, 2], raise an exception
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