import torch
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
import random
from deeprobust.graph.defense import GCN
from deeprobust.graph.global_attack import MetaApprox, Metattack
from deeprobust.graph.utils import *
from deeprobust.graph.data import Dataset, PtbDataset, PrePtbDataset
from models import DGI
import argparse

seed = 2345

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--seed', type=int, default=15, help='Random seed.')
parser.add_argument('--epochs', type=int, default=200,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=16,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--dataset', type=str, default='cora_ml', choices=['cora', 'cora_ml', 'citeseer', 'polblogs', 'pubmed'], help='dataset')
parser.add_argument('--ptb_rate', type=float, default=0.1,  help='pertubation rate')
parser.add_argument('--model', type=str, default='A-Meta-Self',
        choices=['Meta-Self', 'A-Meta-Self', 'Meta-Train', 'A-Meta-Train'], help='model variant')

args = parser.parse_args()

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda")

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if device != 'cpu':
    torch.cuda.manual_seed(args.seed)

def test(adj, features, labels, idx_train, idx_test):
    ''' test on GCN '''

    # if device != torch.device('cpu'):
    adj = adj.to(device)
    features = features.to(device)
    labels = labels.to(device)

    # adj = normalize_adj_tensor(adj)
    model = DGI(features.shape[1], 64, 'prelu', True).to(device)
    model.load_state_dict(torch.load("best_dgi.pkl"))
    # gcn = GCN(nfeat=features.shape[1],
    #           nhid=args.hidden,
    #           nclass=labels.max().item() + 1,
    #           dropout=args.dropout, device=device)
    gcn = model.gcn
    gcn = gcn.to(device)
    gcn.fit(features, adj, labels, idx_train) # train without model picking
    # gcn.fit(features, adj, labels, idx_train, idx_val) # train with validation model picking
    output = gcn(features, adj)
    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()))

    return acc_test.item()


def main():
    data = Dataset(root='data/', name=args.dataset, setting='nettack', seed=15)
    adj, features, labels = data.adj, data.features, data.labels
    idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test
    idx_unlabeled = np.union1d(idx_val, idx_test)
    adj, features, labels = preprocess(adj, features, labels, preprocess_adj=True)
    print(len(adj))
    print('=== testing GCN on original(clean) graph ===')
    test(adj, features, labels, idx_train, idx_test)
    
    data1 = PrePtbDataset(root='data/meta', name=args.dataset, attack_method='meta')
    adj, features, labels = data1.adj,  data.features, data.labels
    idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test
    idx_unlabeled = np.union1d(idx_val, idx_test)
    adj, features, labels = preprocess(adj, features, labels, preprocess_adj=True)
    print(len(adj))
    print('=== testing GCN on noisy(corrupted) graph ===')
    test(adj, features, labels, idx_train, idx_test)

    # # if you want to save the modified adj/features, uncomment the code below
    # model.save_adj(root='./', name=f'mod_adj')
    # model.save_features(root='./', name='mod_features')

if __name__ == '__main__':
    main()

