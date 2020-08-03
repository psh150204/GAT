import torch
import torch.nn as nn 
import torch.nn.functional as F 
from layer import GraphConvolutionLayer, GraphAttentionLayer, SparseGraphConvolutionLayer, SparseGraphAttentionLayer

# TODO step 1.
class GCN(nn.Module):
    def __init__(self, C, H, F, dropout):
        super(GCN, self).__init__()
        self.layer1 = GraphConvolutionLayer(C, H)
        self.layer2 = GraphConvolutionLayer(H, F)
        self.dropout = nn.Dropout(p = dropout)

    def forward(self, x, adj):
        # X : a tensor with size [N, C]
        x = self.dropout(F.relu(self.layer1(x, adj)))
        return self.layer2(x, adj)
    
# TODO step 2.
class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        super(GAT, self).__init__()
        pass

    def forward(self, x, adj):
        pass

# TODO step 3.
class SpGCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(SpGCN, self).__init__()
        pass

    def forward(self, x, adj):
        pass

class SpGAT(nn.Module):
    def __init__(self,nfeat, nhid, nclass, dropout, alpha, nheads):
        super(SpGAT, self).__init__()
        pass

    def forward(self, x, adj):
        pass