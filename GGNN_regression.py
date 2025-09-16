# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import torch
import torch.nn as nn
import networkx as nx
import torch.nn.functional as F
import numpy as np

class AttentionModule(nn.Module):
    def __init__(self, hidden_size):
        super(AttentionModule, self).__init__()
        # Attention weights, one for each node
        self.attention_weights = nn.Linear(hidden_size, 1)

    def forward(self, hidden_states):
        # Calculate attention scores and perform softmax normalization
        attention_scores = self.attention_weights(hidden_states).softmax(dim=1)
        # Apply attention scores to hidden states
        attended_states = attention_scores * hidden_states
        # Sum the weighted hidden states to obtain a comprehensive hidden state representation
        return attended_states.sum(dim=1)


class GRUCell(nn.Module):
    def __init__(self,input_size,hidden_size):
        super(GRUCell,self).__init__()
        self.hidden_size=hidden_size
        
        self.linear_z=nn.Linear(input_size+hidden_size,hidden_size)
        self.linear_r=nn.Linear(input_size+hidden_size,hidden_size)
        self.linear=nn.Linear(input_size+hidden_size,hidden_size)
        
        self._initialization()
        
    def _initialization(self):
        a=-np.sqrt(1/self.hidden_size)
        b=np.sqrt(1/self.hidden_size)
        torch.nn.init.uniform_(self.linear_z.weight,a,b)
        torch.nn.init.uniform_(self.linear_z.bias,a,b)
        torch.nn.init.uniform_(self.linear_r.weight,a,b)
        torch.nn.init.uniform_(self.linear_r.bias,a,b)
        torch.nn.init.uniform_(self.linear.weight,a,b)
        torch.nn.init.uniform_(self.linear.bias,a,b)
        
    def forward(self,input_,hidden_state):
        inputs_state=torch.cat((input_,hidden_state),-1)
        # print(input_.shape)
        # print(inputs_state.shape)
        #z=sigma(W_z*a+U_z*h(t-1))
        update_gate=self.linear_z(inputs_state).sigmoid()
        #r=sigma(W_r*a+U_r*h(t-1))
        reset_gate=self.linear_r(inputs_state).sigmoid()
        #h_hat(t)=tanh(W*a+U*(r×h(t-1)))
        new_hidden_state=self.linear(torch.cat((input_,reset_gate*hidden_state),-1)).tanh()
        #h(t)=(1-z)×h(t-1)+z×h_hat(t)
        output=(1-update_gate)*hidden_state+update_gate*new_hidden_state
        #return output + hidden_state  # Adding residual connection
        return output

class GGNNModel(nn.Module):
    
    def __init__(self,attr_size,hidden_size,propag_steps,output_size):
        super(GGNNModel,self).__init__()
        self.attr_size=attr_size
        self.hidden_size=hidden_size
        self.propag_steps=propag_steps
        self.output_size=output_size
        self.attention_module = AttentionModule(hidden_size)
        self.linear_i=nn.Linear(attr_size,hidden_size)
        self.gru=GRUCell(2*hidden_size,hidden_size)
        self.gru1=GRUCell(2*hidden_size,1)
        self.linear_o=nn.Linear(hidden_size,output_size)
        # self.layer_norm = nn.LayerNorm(hidden_size)  # Add layer normalization
        self._initialization()
        
    def _initialization(self):
        torch.nn.init.kaiming_normal_(self.linear_i.weight)
        torch.nn.init.constant_(self.linear_i.bias,0)
        torch.nn.init.xavier_normal_(self.linear_o.weight)
        torch.nn.init.constant_(self.linear_o.bias,0)
        
    def forward(self,attr_matrix,adj_matrix):
        
        '''
        attr_matrix of shape (batch,graph_size,attributes dimension)
        adj_matrix of shape(batch,graph_size,graph_size)
        The adjacency matrix only has 0 and 1, meaning the edge types are like this
        '''
        # return attr_matrix.squeeze(-1)
        
        A_in=adj_matrix.float()
        #A_in=torch.from_numpy(adj_matrix)
        A_out=A_in.transpose(-1,-2)
        
        # # Add self-connections
        # A_in = A_in + torch.eye(A_in.size(1)).to(A_in.device)
        # A_out = A_out + torch.eye(A_out.size(1)).to(A_out.device)
        if len(A_in.shape)<3:
            A_in=torch.unsqueeze(A_in,0)
            A_out=torch.unsqueeze(A_out,0)
        if len(attr_matrix.shape)<3:
            attr_matrix=torch.unsqueeze(attr_matrix,0)      
        
        hidden_state=self.linear_i(attr_matrix.float()).relu()
        # hidden_state = self.layer_norm(hidden_state)  # Apply layer normalization
        
        for step in range(self.propag_steps):
            # a_v = A_v[h_1 ... h_|V|] (sum aggregation)
            a_in =torch.bmm(A_in,hidden_state)
            a_out=torch.bmm(A_out,hidden_state)            
            hidden_state=self.gru(torch.cat((a_in,a_out),-1),hidden_state)    
         
        # For regression, no need to consider the final normalization issue
        output=self.linear_o(hidden_state)
        return output



