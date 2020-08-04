import torch
import torch.nn as nn 
import torch.nn.functional as F 
from layer import GraphConvolutionLayer, GraphAttentionLayer

class GCN(nn.Module):
    def __init__(self, F, H, C, dropout):
        super(GCN, self).__init__()
        self.layer1 = GraphConvolutionLayer(F, H)
        self.layer2 = GraphConvolutionLayer(H, C)
        self.dropout = nn.Dropout(p = dropout)

    def forward(self, x, adj):
        # X : a tensor with size [N, F]
        
        x = self.dropout(F.relu(self.layer1(x, adj))) # [N, H]
        return self.layer2(x, adj) # [N, C]
    
class GAT(nn.Module):
    def __init__(self, F, H, C, dropout, alpha, K):
        super(GAT, self).__init__()
        self.layer1 = GraphAttentionLayer(F, H, K, alpha)
        self.layer2 = GraphAttentionLayer(K * H, C, 1, alpha, concat = False)
        self.dropout = nn.Dropout(p = dropout)

    def forward(self, x, adj):
        # x : a tensor with size [N, F]

        x = self.dropout(F.relu(self.layer1(x, adj))) # [N, KH]
        return self.layer2(x, adj) # [N, C]