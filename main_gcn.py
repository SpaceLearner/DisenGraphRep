import torch
import torch.nn as nn
import torch.nn.functional

from deeprobust.graph.defense import GCN
from deeprobust.graph.global_attack import MetaApprox, Metattack
from deeprobust.graph.utils import *
from deeprobust.graph.data import Dataset, PtbDataset

from torch_geometric.nn import DenseGCNConv

data = Dataset(root='/tmp/', name='cora', setting='nettack')
adj1, features1, labels1 = data.adj, data.features, data.labels
idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test
idx_unlabeled = np.union1d(idx_val, idx_test)

perturbed_data = PtbDataset(root='/tmp/', name='cora')
perturbed_adj = perturbed_data.adj

# perturbations = int(args.ptb_rate * (adj.sum()//2))
adj, features, labels = preprocess(adj1, features1, labels1, preprocess_adj=False)
perturbed_adj, features, labels = preprocess(perturbed_adj, features1, labels1, preprocess_adj=False)

# model = DenseGCNConv(features.shape[1], max(labels)+1)

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        print(features.shape, labels.shape)
        self.conv1 = DenseGCNConv(features.shape[1], 16)
        self.conv2 = DenseGCNConv(16, int(max(labels)+1))
        # self.conv1 = ChebConv(data.num_features, 16, K=2)
        # self.conv2 = ChebConv(16, data.num_features, K=2)

    def forward(self, x, adj):
        x, adj = x.unsqueeze(0), adj.unsqueeze(0)
        x = F.relu(self.conv1(x, adj))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, adj)
        return F.log_softmax(x, dim=1)
    
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')
model, features, labels, adj = Net().to(device), features.to(device), labels.to(device), adj.to(device)
optimizer = torch.optim.Adam([
    dict(params=model.conv1.parameters(), weight_decay=5e-4),
    dict(params=model.conv2.parameters(), weight_decay=0)
], lr=0.01)  # Only perform weight-decay on first convolution.


def train():
    model.train()
    optimizer.zero_grad()
    F.nll_loss(model(features, adj)[0, idx_train], labels[idx_train]).backward()
    optimizer.step()


@torch.no_grad()
def test():
    model.eval()
    logits, accs = model(features, perturbed_adj)[0], []
    for  mask in [idx_train, idx_val, idx_test]:
        pred = logits[mask].max(1)[1]
        acc = pred.eq(labels[mask]).sum().item() / len(mask)
        accs.append(acc)
    return accs


best_val_acc = test_acc = 0
for epoch in range(1, 201):
    train()
    train_acc, val_acc, tmp_test_acc = test()
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        test_acc = tmp_test_acc
    log = 'Epoch: {:03d}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}'
    print(log.format(epoch, train_acc, best_val_acc, test_acc))