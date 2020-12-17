import math
import torch
from torch.nn import Parameter
from torch.autograd import Variable
from numbers import Number
from sklearn.linear_model import LogisticRegression
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

class DeepGraphInfomax(torch.nn.Module):
    r"""The Deep Graph Infomax model from the
    `"Deep Graph Infomax" <https://arxiv.org/abs/1809.10341>`_
    paper based on user-defined encoder and summary model :math:`\mathcal{E}`
    and :math:`\mathcal{R}` respectively, and a corruption function
    :math:`\mathcal{C}`.

    Args:
        hidden_channels (int): The latent space dimensionality.
        encoder (Module): The encoder module :math:`\mathcal{E}`.
        summary (callable): The readout function :math:`\mathcal{R}`.
        corruption (callable): The corruption function :math:`\mathcal{C}`.
    """

    def __init__(self, hidden_channels, encoder, summary, corruption):
        super(DeepGraphInfomax, self).__init__()
        self.hidden_channels = hidden_channels
        self.z_dim = self.hidden_channels
        self.encoder = encoder
        self.summary = summary
        self.corruption = corruption

        self.weight = Parameter(torch.Tensor(hidden_channels, hidden_channels))

        self.x_dist = dist.Bernoulli()
        self.q_dist = dist.Normal()
        self.prior_dist = dist.Normal()
        
        self.register_buffer('prior_params', torch.zeros(self.hidden_channels, 2))
        
        self.beta = 1
        self.include_mutinfo = True
        self.lamb = 0

        self.reset_parameters()

    def reset_parameters(self):
        reset(self.encoder)
        reset(self.summary)
        uniform(self.hidden_channels, self.weight)

    def forward(self, *args, **kwargs):
        """Returns the latent space for the input arguments, their
        corruptions and their summary representation."""
        pos_z_params = self.encoder(*args, **kwargs).view(args[0].size(0), self.hidden_channels, self.q_dist.nparams)
        pos_z = self.q_dist.sample(params=pos_z_params)
        
        cor = self.corruption(*args, **kwargs)
        cor = cor if isinstance(cor, tuple) else (cor, )
        neg_z_params = self.encoder(*cor).view(args[0].size(0), self.hidden_channels, self.q_dist.nparams)
        neg_z = self.q_dist.sample(params=neg_z_params)
        
        # pos_z = pos_z_params.view(args[0].size(0), -1)
        # neg_z = neg_z_params.view(args[0].size(0), -1)
        
        summary = self.summary(pos_z, *args, **kwargs)
        elbo2 = self.elbo2(args[0], args[1], pos_z.size(0))
        
        return pos_z, neg_z, summary, elbo2
    
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

    def elbo2(self, x, edge_index, n_nodes):
        
        batch_size = x.size(0)
        prior_params = self._get_prior_params(batch_size)
        
        z_params = self.encoder(x, edge_index).view(x.size(0), self.hidden_channels, self.q_dist.nparams)
        zs = self.q_dist.sample(params=z_params)
        
        # x_params = self.decoder(zs, edge_index).view(x.size(0), 1, -1)
        # xs = self.x_dist.sample(params=x_params)
        # logpx = self.x_dist.log_density(x, params=x_params).view(batch_size, -1).sum(1)
        logpz = self.prior_dist.log_density(zs, params=prior_params).view(batch_size, -1).sum(1)
        logqz_condx = self.q_dist.log_density(zs, params=z_params).view(batch_size, -1).sum(1)
        
        elbo2 = logpz - logqz_condx
        
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
                modified_elbo = self.beta * (
                    (logqz_condx - logpz) -
                    self.lamb * (logqz_prodmarginals - logpz)
                )
            else:
                modified_elbo = self.beta * (
                    (logqz - logqz_prodmarginals) +
                    (1 - self.lamb) * (logqz_prodmarginals - logpz)
                )
        else:
            if self.include_mutinfo:
                modified_elbo =  (logqz_condx - logqz) - \
                    self.beta * (logqz - logqz_prodmarginals) - \
                    (1 - self.lamb) * (logqz_prodmarginals - logpz)
            else:
                modified_elbo = self.beta * (logqz - logqz_prodmarginals) - \
                    (1 - self.lamb) * (logqz_prodmarginals - logpz)

        return modified_elbo, elbo2.detach()
        

    def discriminate(self, z, summary, sigmoid=True):
        r"""Given the patch-summary pair :obj:`z` and :obj:`summary`, computes
        the probability scores assigned to this patch-summary pair.

        Args:
            z (Tensor): The latent space.
            sigmoid (bool, optional): If set to :obj:`False`, does not apply
                the logistic sigmoid function to the output.
                (default: :obj:`True`)
        """
        value = torch.matmul(z, torch.matmul(self.weight, summary))
        return torch.sigmoid(value) if sigmoid else value

    def loss(self, pos_z, neg_z, summary, elbo2):
        r"""Computes the mutal information maximization objective."""
        pos_loss = -torch.log(
            self.discriminate(pos_z, summary, sigmoid=True) + EPS).mean()
        neg_loss = -torch.log(
            1 - self.discriminate(neg_z, summary, sigmoid=True) + EPS).mean()
        
        return pos_loss + neg_loss - 2.0 * elbo2[0].mean()

    def test(self, train_z, train_y, test_z, test_y, solver='lbfgs',
             multi_class='auto', *args, **kwargs):
        r"""Evaluates latent space quality via a logistic regression downstream
        task."""
        clf = LogisticRegression(solver=solver, multi_class=multi_class, *args,
                                 **kwargs).fit(train_z.detach().cpu().numpy(),
                                               train_y.detach().cpu().numpy())
        return clf.score(test_z.detach().cpu().numpy(),
                         test_y.detach().cpu().numpy())

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, self.hidden_channels)