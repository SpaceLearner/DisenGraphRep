import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from layers import GCN, AvgReadout, Discriminator, GCNt
import models.dist as dist
import math

EPS = 1e-15

def uniform(size, tensor):
    bound = 1.0 / math.sqrt(size)
    if tensor is not None:
        tensor.data.uniform_(-bound, bound)
        
def reset(nn):
    def _reset(item):
        if hasattr(item, 'reset_parameters'):
            item.reset_parameters()

    if nn is not None:
        if hasattr(nn, 'children') and len(list(nn.children())) > 0:
            for item in nn.children():
                _reset(item)
        else:
            _reset(nn)
            
def logsumexp(value, dim=None, keepdim=False):
    """Numerically stable implementation of the operation

    value.exp().sum(dim, keepdim).log()
    """
    if dim is not None:
        m, _ = torch.max(value, dim=dim, keepdim=True)
        value0 = value - m
        if keepdim is False:
            m = m.squeeze(dim)
        return m + torch.log(torch.sum(torch.exp(value0),
                                       dim=dim, keepdim=keepdim))
    else:
        m = torch.max(value)
        sum_exp = torch.sum(torch.exp(value - m))
        if isinstance(sum_exp, Number):
            return m + math.log(sum_exp)
        else:
            return m + torch.log(sum_exp)
        
def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

class DGI_BW(nn.Module):
    def __init__(self, n_in, n_h, activation, cuda):
        super(DGI_BW, self).__init__()
        # self.gcn = GCN(n_in, n_h, activation)
        self.device = torch.device('cuda' if torch.cuda.is_available() and cuda else 'cpu')
        self.gcn = GCNt(n_in, n_h, 7, device=self.device)
        self.decoder = GCN(n_h, n_in, act=F.relu)
        self.n_h = n_h
        self.n_in = n_in
        self.z_dim = self.n_h
        self.proj = nn.Sequential(nn.Linear(self.n_h, self.n_h), nn.BatchNorm1d(self.n_h), nn.ReLU(), nn.Linear(self.n_h, self.n_h))
        self.read = AvgReadout()

        self.sigm = nn.Sigmoid()
        
        self.lambd = 3.9e-3
        
        self.scale_loss = 1 / 32

        self.disc = Discriminator(n_h)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        for weight in self.parameters():
            if len(weight.shape) > 2:
                nn.init.xavier_uniform_(weight)
        
    def forward(self, seq1, seq2, adj, sparse, msk, samp_bias1, samp_bias2):
        h_1 = self.gcn.forward1(seq1, adj, sparse).view(-1, self.n_h)

        c = self.read(h_1.unsqueeze(0), msk)
        c = self.sigm(c)

        h_2 = self.gcn.forward1(seq2, adj, sparse).view(-1, self.n_h)
        # print(c.shape, h_1.shape, h_2.shape)
        ret = self.disc(c, h_1.unsqueeze(0), h_2.unsqueeze(0), samp_bias1, samp_bias2)

        return ret
    
    def bw_loss(self, feat):
        feat = feat.squeeze()
        feat = self.proj(feat)
        
        c = feat.T @ feat
        
        c.div_(len(feat))
        # torch.distributed.all_reduce(c)
        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum().mul(self.scale_loss)
        off_diag = off_diagonal(c).pow_(2).sum().mul(self.scale_loss)
        loss = on_diag + self.lambd * off_diag
        
        return loss
        

    # Detach the return variables
    def embed(self, seq, adj, sparse, msk):
        h_1 = self.gcn.forward1(seq, adj, sparse).view(-1, self.n_h)
        c = self.read(h_1.unsqueeze(0), msk)

        return h_1.detach(), c.detach()