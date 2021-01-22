import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from layers import GCN, AvgReadout, Discriminator, GCNt
import models.dist as dist

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

class DGI(nn.Module):
    def __init__(self, n_in, n_h, activation, cuda):
        super(DGI, self).__init__()
        # self.gcn = GCN(n_in, n_h, activation)
        self.device = torch.device('cuda' if torch.cuda.is_available() and cuda else 'cpu')
        self.gcn = GCNt(n_in, 2 * n_h, 7, device=self.device)
        self.decoder = GCN(n_h, n_in, act=F.relu)
        self.n_h = n_h
        self.n_in = n_in
        self.z_dim = self.n_h
        self.read = AvgReadout()

        self.sigm = nn.Sigmoid()
        
        self.x_dist = dist.Bernoulli()
        self.q_dist = dist.Normal()
        self.prior_dist = dist.Normal()
        
        self.register_buffer('prior_params', torch.zeros(n_h, 2).to(self.device))
        
        self.beta = 1
        self.include_mutinfo = True
        self.lamb = 0
        self.mss = True

        self.disc = Discriminator(n_h)

    def forward(self, seq1, seq2, adj, sparse, msk, samp_bias1, samp_bias2):
        h_1 = self.gcn.forward1(seq1, adj, sparse).view(-1, self.n_h, self.q_dist.nparams)
        h_1 = h_1[:, :, 0]

        c = self.read(h_1.unsqueeze(0), msk)
        c = self.sigm(c)

        h_2 = self.gcn.forward1(seq2, adj, sparse).view(-1, self.n_h, self.q_dist.nparams)
        h_2 = h_2[:, :, 0]
        # print(c.shape, h_1.shape, h_2.shape)
        ret = self.disc(c, h_1.unsqueeze(0), h_2.unsqueeze(0), samp_bias1, samp_bias2)

        return ret

    # Detach the return variables
    def embed(self, seq, adj, sparse, msk):
        h_1 = self.gcn.forward1(seq, adj, sparse).view(-1, self.n_h, self.q_dist.nparams)[:, :, 0]
        c = self.read(h_1.unsqueeze(0), msk)

        return h_1.detach(), c.detach()
    
    def _get_prior_params(self, batch_size=1):
        expanded_size = (batch_size,) + self.prior_params.size()
        prior_params = Variable(self.prior_params.expand(expanded_size))
        return prior_params
    
    def _log_importance_weight_matrix(self, batch_size, dataset_size):
        N = dataset_size
        M = batch_size - 1
        strat_weight = (N - M) / (N * M)
        W = torch.Tensor(batch_size, batch_size).fill_(1 / M)
        W.view(-1)[::M+1] = 1 / N
        W.view(-1)[1::M+1] = strat_weight
        W[M-1, 0] = strat_weight
        return W.log()

    def elbo2(self, x, adj, n_nodes):
        
        batch_size = x.size(1)
        prior_params = self._get_prior_params(batch_size)
        
        z_params = self.gcn.forward1(x, adj.unsqueeze(0))
        
        z_params = z_params.view(x.size(1), self.n_h, self.q_dist.nparams)
        zs = self.q_dist.sample(params=z_params)

        x_params = self.decoder(zs.unsqueeze(0), adj.unsqueeze(0)).view(x.size(1), 1, -1)
        xs = self.x_dist.sample(params=x_params)
        # print(x.shape)
        logpx = self.x_dist.log_density(x.squeeze().unsqueeze(1), params=x_params).view(batch_size, -1).sum(1)
        logpz = self.prior_dist.log_density(zs, params=prior_params).view(batch_size, -1).sum(1)
        logqz_condx = self.q_dist.log_density(zs, params=z_params).view(batch_size, -1).sum(1)
        
        elbo2 = logpx + logpz - logqz_condx
        
        if self.beta == 1 and self.include_mutinfo and self.lamb == 0:
            return elbo2, elbo2.detach()

        # compute log q(z) ~= log 1/(NM) sum_m=1^M q(z|x_m) = - log(MN) + logsumexp_m(q(z|x_m))
        _logqz = self.q_dist.log_density(
            zs.view(batch_size, 1, self.z_dim),
            z_params.view(1, batch_size, self.z_dim, self.q_dist.nparams)
        )

        if not self.mss:
            # minibatch weighted sampling
            logqz_prodmarginals = (logsumexp(_logqz, dim=1, keepdim=False) - math.log(batch_size * n_nodes)).sum(1)
            logqz = (logsumexp(_logqz.sum(2), dim=1, keepdim=False) - math.log(batch_size * n_nodes))
        else:
            # minibatch stratified sampling
            logiw_matrix = Variable(self._log_importance_weight_matrix(batch_size, n_nodes).type_as(_logqz.data))
            logqz = logsumexp(logiw_matrix + _logqz.sum(2), dim=1, keepdim=False)
            logqz_prodmarginals = logsumexp(
                logiw_matrix.view(batch_size, batch_size, 1) + _logqz, dim=1, keepdim=False).sum(1)

        if not self.tcvae:
            if self.include_mutinfo:
                modified_elbo = logpx - self.beta * (
                    (logqz_condx - logpz) -
                    self.lamb * (logqz_prodmarginals - logpz)
                )
            else:
                modified_elbo = logpx - self.beta * (
                    (logqz - logqz_prodmarginals) +
                    (1 - self.lamb) * (logqz_prodmarginals - logpz)
                )
        else:
            if self.include_mutinfo:
                modified_elbo =  logpx - (logqz_condx - logqz) - \
                    self.beta * (logqz - logqz_prodmarginals) - \
                    (1 - self.lamb) * (logqz_prodmarginals - logpz)
            else:
                modified_elbo = logpx - self.beta * (logqz - logqz_prodmarginals) - \
                    (1 - self.lamb) * (logqz_prodmarginals - logpz)

        return modified_elbo, elbo2.detach()