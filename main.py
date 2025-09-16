import argparse
import os
import numpy as np
import torch
import GGNN_regression
import utils_regression



# Define arguments
parser = argparse.ArgumentParser()
parser.add_argument('--train_number',type=int,default=5,help='test data location')
parser.add_argument('--duration',type=int,default=12,help='data collection time(h)')
parser.add_argument('--output_size',type=int,default=12,help='the length of the prediction')
parser.add_argument('--network_name', type=str, required=True, help='name of the network') 
parser.add_argument('--weight_type', type=str, default='diameter_length', help='length,diameter_length,binary') 
parser.add_argument('--weight_direction', type=str, default='directed', help='directed,undirected') 
parser.add_argument('--init', action='store_true', help='Specify if initialization is needed')
parser.add_argument('--distance_type', type=str, default='binary', help='length,diameter_length,binary') 
parser.add_argument('--distance_direction', type=str, default='undirected', help='directed,undirected') 
#parser.add_argument('--adj_name', type=str, default='topo', help='type of node adjacency matrix')
parser.add_argument('--loss_type', type=int, default=0, help='loss type:0:base loss,1: base +lower extral loss,2: base +upper extral loss')
parser.add_argument('--percent', type=float, required=True, help='percent value')
parser.add_argument('--epochs', type=int, default=40, help='number of epochs to train')
parser.add_argument('--batch_size', type=int, default=32, help='batch size')
parser.add_argument('--propag_steps', type=int, default=15, help='number of propagation steps')
parser.add_argument('--hidden_size', type=int, default=128, help='number of hidden units')
parser.add_argument('--standard_type',type=int,default=1,help='0:不归一化，1：全局归一化，2：分区归一化')
parser.add_argument('--threshold',type=float,default=0.0001,help='physics threshold')
parser.add_argument('--w',type=float,default=1,help='physics 权重')

#parser.add_argument('--num_head', type=int, default=1, help='number of attention head')
#parser.add_argument('--sensor_number', type=int, default=33, help='number of sensors')
# parser.add_argument('--mask_0_file', type=str, default='./data_generate/dataset_all_new.npz', help='path to mask 0 file')
# parser.add_argument('--mask_2_file', type=str, default='./data_generate/dataset_all_new.npz', help='path to mask 2 file')
# parser.add_argument('--pred_bounds_file', type=str, default='./data_generate/dataset_all_new.npz', help='path to pred_bounds file')
# parser.add_argument('--node_adj_file', type=str, default='./data_generate/node_adj_uni.npy', help='path to node adjacency matrix file')
# parser.add_argument('--geo_adj_file', type=str, default='./data_generate/geo_adj_uni.npy', help='path to geographical adjacency matrix file')
#parser.add_argument('--feature_size', type=int, default=24, help='2h,5min interval')
args = parser.parse_args()
print('掩膜:{},持续时间：{}h,输出时间：{}h，回合数：{}，损失类型：{}，标准化类型：{}，隐藏状态数：{}，批样本数：{}，传播步：{}'.format(
  args.percent,args.duration,args.output_size,args.epochs,args.loss_type,args.standard_type,args.hidden_size,args.batch_size,args.propag_steps))
# Load data
#将原始的数据进行一个数据的转化(得到的是原始的水质数据：all_source)
# Load data
data = f'./{args.network_name}/data_generate/dataset_all_new1.npz'  # 使用 network_name 作为文件路径的一部分
all_source= np.load(data)['dataset']
#设置一个n_frame直接是1,2,4,6,12
#n_frame=int(args.duration*60/args.report_time+1)   #历史数据+预测步长数据    
n_frame=args.duration+args.output_size   
n_slot=all_source.shape[0]-n_frame+1
tmp_seq=np.zeros((n_slot,n_frame,all_source.shape[1]))
for i in range(n_slot):
    sta=i
    end=sta+n_frame
    tmp_seq[i,:,:]=all_source[sta:end,:]   
    
#下面这个是最初的也就是最后一份是测试集的情况         
all_source_1=all_source[n_frame-1:]
train_len=int(all_source_1.shape[0]*0.8)
train_input,test_input=tmp_seq[:train_len],tmp_seq[train_len:]

#下面这个是将测试集分成5分，每一份都作为一次测试集的情况
# 对于每个部分，将它作为测试集，其他部分作为训练集
n_samples = tmp_seq.shape[0]
n_folds = 5
fold_size = n_samples // n_folds
indices = np.arange(n_samples)
split_indices = [indices[i * fold_size:(i + 1) * fold_size] for i in range(n_folds)]
i=args.train_number-1
test_index = split_indices[i]
train_index = np.concatenate([split_indices[j] for j in range(n_folds) if j != i])
train_input = tmp_seq[train_index]
test_output = tmp_seq[test_index]

mask_0 = np.load(data)['mask_measure']
mask_1 = np.load(data)['mask_sample']
mask_2 = np.load(data)['mask_prediction']
if args.network_name == 'L_town':
  mask_area=np.load(data)['mask_area']
  mask_area_tensor = torch.tensor(mask_area) 
else:
  mask_area_tensor =1#这个地方是瞎填的，也就是其他的管网不需要管他

node_adj = np.load(f'./{args.network_name}/data_generate/adj_matrices_link_weight_{args.weight_type}_{args.weight_direction}.npy', allow_pickle=True)
geo_adj = np.load(f'./{args.network_name}/data_generate/adj_matrices_link_distance_{args.distance_type}_{args.distance_direction}.npy', allow_pickle=True)
#print(node_adj) 
steps_per_epoch= int(len(train_input)/args.batch_size)

feature_size=args.duration

model,opt, stat = utils_regression.train_model(GGNN_regression.GGNNModel(feature_size, args.hidden_size, args.propag_steps,args.output_size).cuda(), 
                                               args.batch_size, feature_size,args.output_size,args.epochs,steps_per_epoch, train_input, test_input, 
                                               node_adj, geo_adj, mask_0, mask_1, mask_2, mask_area_tensor,yanmo=True, 
                                               percent=args.percent,loss_type=args.loss_type,standard_type=args.standard_type,threshold=args.threshold,init=args.init,w=args.w)
output_dir = f'./{args.network_name}/model' 
if not os.path.exists(output_dir):
  os.makedirs(output_dir)

weight_name=f'weight_{args.weight_type}_{args.weight_direction}_distance_{args.distance_type}_{args.distance_direction}'
test_dir = os.path.join(output_dir, f'mask{args.percent}_new{args.train_number}_{args.duration}h_{args.output_size}_standard{args.standard_type}_{weight_name}')  
if not os.path.exists(test_dir):
  os.makedirs(test_dir)
    
file_name=test_dir+'/PRO'+str(args.propag_steps)+'_'+str(args.epochs)+'E_'+str(args.batch_size)+'BA'+str(args.hidden_size)+'loss'+str(args.loss_type)+'_w'+str(args.w)
if args.init:  
  torch.save(model.state_dict(),file_name+'_model_state_dict.pt')
  torch.save(opt.state_dict(),file_name+'_optimizer_state_dict.pt')
  torch.save(stat,file_name+'_training_statistics.pt')
else:
  torch.save(model.state_dict(),file_name+'_model_state_dict_noninit.pt')
  torch.save(opt.state_dict(),file_name+'_optimizer_state_dict_noninit.pt')
  torch.save(stat,file_name+'_training_statistics_noninit.pt')