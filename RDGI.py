import math

import torch
from torch.nn import Parameter
import torch.nn.functional as F
import torch.optim as optim
from sklearn.linear_model import LogisticRegression

from deeprobust.graph import utils

from copy import deepcopy

def uniform(size, tensor):
    if tensor is not None:
        bound = 1.0 / math.sqrt(size)
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


EPS = 1e-15


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
        self.encoder = encoder
        self.summary = summary
        self.corruption = corruption

        self.optimizer = optim.Adam(self.parameters(), weight_decay=5e-4)

        self.weight = Parameter(torch.Tensor(hidden_channels, hidden_channels))

        self.clf = LogisticRegression(solver='lbfgs', multi_class='auto',*args, **kwargs)

        self.reset_parameters()

    def reset_parameters(self):
        reset(self.encoder)
        reset(self.summary)
        uniform(self.hidden_channels, self.weight)

    def fit(self, features, adj, labels, idx_train, idx_val=None, train_iters=200, initialize=True, verbose=True, normalize=True, patience=500, **kwargs):
        
        self.device = self.gc1.weight.device
        if initialize:
            self.initialize()

        if type(adj) is not torch.Tensor:
            features, adj, labels = utils.to_tensor(features, adj, labels, device=self.device)
        else:
            features = features.to(self.device)
            adj = adj.to(self.device)
            labels = labels.to(self.device)

        if normalize:
            if utils.is_sparse_tensor(adj):
                adj_norm = utils.normalize_adj_tensor(adj, sparse=True)
            else:
                adj_norm = utils.normalize_adj_tensor(adj)
        else:
            adj_norm = adj

        self.adj_norm = adj_norm
        self.features = features
        self.labels = labels

        early_stopping = 50
        best_loss_val = 100

        for i in range(train_iters):
            self.train()
            optimizer.zero_grad()
            output = self(features, adj_norm)
            loss_train = self.loss(output)
            loss_train.backward()
            optimizer.step()

            if verbose and i % 10 == 0:
                print('Epoch {}, training loss: {}'.format(i, loss_train.item()))

            self.eval()

            loss_val = loss_train

            if best_loss_val > loss_val:
                best_loss_val = loss_val
                self.output = output
                weights = deepcopy(self.state_dict())
                patience = early_stopping
            else:
                patience -= 1
            if i > early_stopping and patience <= 0:
                break

        if verbose:
             print('=== early stopping at {0}, loss_val = {1} ==='.format(i, best_loss_val) )
        self.load_state_dict(weights)
        
        embeds = self.get_embeds(features, adj_norm).detach()
        
        X = embeds[idx_train]
        y = labels[idx_train]
        
        self.clf.fit(X, y)
        

    def forward(self, *args, **kwargs):
        """Returns the latent space for the input arguments, their
        corruptions and their summary representation."""
        pos_z = self.encoder(*args, **kwargs)
        cor = self.corruption(*args, **kwargs)
        cor = cor if isinstance(cor, tuple) else (cor, )
        neg_z = self.encoder(*cor)
        summary = self.summary(pos_z, *args, **kwargs)
        return pos_z, neg_z, summary
    
    def get_embeds(self, features, adj):

        if features is None:
            features = self.features
        if adj is None:
            adj = self.adj
        
        embeds = self.encoder(features, adj).detach().numpy()
        
        return embeds

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

    def loss(self, pos_z, neg_z, summary):
        r"""Computes the mutual information maximization objective."""
        pos_loss = -torch.log(
            self.discriminate(pos_z, summary, sigmoid=True) + EPS).mean()
        neg_loss = -torch.log(
            1 - self.discriminate(neg_z, summary, sigmoid=True) + EPS).mean()

        return pos_loss + neg_loss

    def predict(features=None, adj=None, *args, **kwargs):

        self.eval()
        
        if features is None and adj is None:
            embeds = self.get_embeds(self.features, self.adj).detach().numpy()
            return self.clf.predict_log_proba(embeds)
        else:
            if type(adj) is not torch.Tensor:
                features, adj = utils.to_tensor(features, adj, device=self.device)

            self.features = features
            if utils.is_sparse_tensor(adj):
                self.adj_norm = utils.normalize_adj_tensor(adj, sparse=True)
            else:
                self.adj_norm = utils.normalize_adj_tensor(adj)
            
            embeds = self.get_embeds(self.features, self.adj_norm)
            
            return self.clf.predict_log_proba(embeds)
            
    def test(self, idx_test):
        """Evaluate GCN performance on test set.

        Parameters
        ----------
        idx_test :
            node testing indices
        """
        self.eval()
        output = self.predict()
        # output = self.output
        loss_test = F.nll_loss(output[idx_test], self.labels[idx_test])
        acc_test = utils.accuracy(output[idx_test], self.labels[idx_test])
        print("Test set results:",
              "loss= {:.4f}".format(loss_test.item()),
              "accuracy= {:.4f}".format(acc_test.item()))
        return acc_test.item()

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, self.hidden_channels)