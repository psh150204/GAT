
import numpy as np 
import torch 
import torch.nn as nn 
import torch.nn.functional as F 

class GraphConvolutionLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(GraphConvolutionLayer,self).__init__()
        self.W = nn.Parameter(torch.zeros(in_features, out_features, dtype = torch.float32))
        nn.init.xavier_uniform_(self.W) # initialize as described in Glorot & Bengio (2010)
    
    def forward(self, input, adj):
        # input (= X) : a tensor with size [N, F]
        # adj (= A_hat) : a tensor with size [N, N]

        return torch.mm(adj, torch.mm(input, self.W))

class Attention(nn.Module):
    # single head attention
    def __init__(self, in_features, out_features, alpha):
        super(Attention, self).__init__()
        self.alpha = alpha

        self.W = nn.Linear(in_features, out_features, bias = False)
        self.a_T = nn.Linear(2 * out_features, 1, bias = False)

        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.a_T.weight)

    def forward(self, h, adj):
        # h : a tensor with size [N, F] where N be a number of nodes and F be a number of features
        N = h.size(0)
        Wh = self.W(h) # h -> Wh : [N, F] -> [N, F']
        
        # H1 : [N, N, F'], H2 : [N, N, F'], attn_input = [N, N, 2F']

        # H1 = [[h1 h1 ... h1]   |  H2 = [[h1 h2 ... hN]   |   attn_input = [[h1||h1 h1||h2 ... h1||hN]
        #       [h2 h2 ... h2]   |        [h1 h2 ... hN]   |                 [h2||h1 h2||h2 ... h2||hN]
        #            ...         |             ...         |                         ...
        #       [hN hN ... hN]]  |        [h1 h2 ... hN]]  |                 [hN||h1 hN||h2 ... hN||hN]]
        
        H1 = Wh.unsqueeze(1).repeat(1,N,1)
        H2 = Wh.unsqueeze(0).repeat(N,1,1)
        attn_input = torch.cat([H1, H2], dim = -1)

        e = F.leaky_relu(self.a_T(attn_input).squeeze(-1), negative_slope = self.alpha) # [N, N]
        
        attn_mask = -1e18*torch.ones_like(e)
        masked_e = torch.where(adj > 0, e, attn_mask)
        attn_scores = F.softmax(masked_e, dim = -1) # [N, N]

        h_prime = torch.mm(attn_scores, Wh) # [N, F']

        return F.elu(h_prime) # [N, F']

class GraphAttentionLayer(nn.Module):
    # multi head attention
    def __init__(self, in_features, out_features, num_heads, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.concat = concat
        self.attentions = nn.ModuleList([Attention(in_features, out_features, alpha) for _ in range(num_heads)])
        
    def forward(self, input, adj):
        # input (= X) : a tensor with size [N, F]

        if self.concat :
            # concatenate
            outputs = []
            for attention in self.attentions:
                outputs.append(attention(input, adj))
            
            return torch.cat(outputs, dim = -1) # [N, KF']

        else :
            # average
            output = None
            for attention in self.attentions:
                if output == None:
                    output = attention(input, adj)
                else:
                    output += attention(input, adj)
            
            return output/len(self.attentions) # [N, F']