from torch_geometric.datasets import Planetoid, Amazon, Coauthor, Flickr, Reddit2, Reddit
import numpy as np
import torch
from torch_sparse import spmm
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_sparse import SparseTensor, fill_diag, sum, mul
from torch_geometric.utils import add_remaining_self_loops,add_self_loops
from torch_scatter import scatter_add
from ogb.nodeproppred import PygNodePropPredDataset, Evaluator
import torch_geometric.transforms as T
import os

def gcn_norm(edge_index, edge_weight=None, num_nodes=None, improved=False,
             add_self_loops=True, dtype=None):
    fill_value = 2. if improved else 1.

    if isinstance(edge_index, SparseTensor):
        adj_t = edge_index
        if not adj_t.has_value():
            adj_t = adj_t.fill_value(1., dtype=dtype)
        if add_self_loops:
            adj_t = fill_diag(adj_t, fill_value)
        deg = sum(adj_t, dim=1)
        deg_inv_sqrt = deg.pow_(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0.)
        adj_t = mul(adj_t, deg_inv_sqrt.view(-1, 1))
        adj_t = mul(adj_t, deg_inv_sqrt.view(1, -1))
        return adj_t

    else:
        num_nodes = maybe_num_nodes(edge_index, num_nodes)

        if edge_weight is None:
            edge_weight = torch.ones((edge_index.size(1),), dtype=dtype,
                                     device=edge_index.device)

        if add_self_loops:
            edge_index, tmp_edge_weight = add_remaining_self_loops(
                edge_index, edge_weight, fill_value, num_nodes)
            assert tmp_edge_weight is not None
            edge_weight = tmp_edge_weight

        row, col = edge_index[0], edge_index[1]
        deg = scatter_add(edge_weight, col, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow_(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
        # return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]
        edge_weight = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]
        n = torch.max(edge_index[0]) + 1  # num of nodes
        adj_norm = SparseTensor(row=edge_index[0], col=edge_index[1], value=edge_weight,
                                sparse_sizes=(n, n), is_sorted=True)
        return adj_norm


def onehot(label):
    """
    return the onehot label for mse loss
    """
    classes = set(label.detach().numpy())
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
    x = list(map(classes_dict.get, label.detach().numpy()))
    label_onehot = np.array(x)
    label_onehot = torch.tensor(label_onehot, dtype=torch.float)
    return label_onehot


def az(adj, z):
    if isinstance(adj, SparseTensor):
        row, col, adj_value = adj.coo()
        edge_index = torch.stack([row, col], dim=0)
        row_size = adj.size(dim=0)
        col_size = adj.size(dim=1)
        return spmm(edge_index, adj_value, row_size, col_size, z)


class cora():
    def __init__(self):
        self.data = Planetoid(root='./dataset/cora', name='cora')[0]
        self.dataset = Planetoid(root='./dataset/cora', name='cora')[0]
        self.label = self.data.y
        self.data.y = onehot(self.data.y)
        self.num_classes = self.data.y.size()[1]
        self.y_train, self.y_test, self.y_val = self.data.y[self.data.train_mask], self.data.y[self.data.test_mask], self.data.y[self.data.val_mask]
        self.adj = gcn_norm(self.data.edge_index)
        data_augmentation(self,'cora')
        self.x_train, self.x_test, self.x_val = self.x[self.data.train_mask], self.x[self.data.test_mask], self.x[self.data.val_mask]
        self.train_mask, self.test_mask, self.val_mask =self.data.train_mask, self.data.test_mask, self.data.val_mask


class pubmed():
    def __init__(self):
        self.data = Planetoid(root='./dataset/PubMed', name='PubMed')[0]
        self.dataset = Planetoid(root='./dataset/PubMed', name='PubMed')
        self.label =self.data.y
        self.data.y = onehot(self.data.y)
        self.num_classes = self.data.y.size()[1]
        self.y_train, self.y_test, self.y_val = self.data.y[self.data.train_mask], self.data.y[self.data.test_mask], self.data.y[self.data.val_mask]
        self.adj = gcn_norm(self.data.edge_index)
        data_augmentation(self,'pubmed')
        self.x_train, self.x_test, self.x_val = self.x[self.data.train_mask], self.x[self.data.test_mask], self.x[
            self.data.val_mask]
        self.train_mask, self.test_mask, self.val_mask = self.data.train_mask, self.data.test_mask, self.data.val_mask


class citeseer():
    def __init__(self):
        self.data = Planetoid(root='./dataset/citeseer', name='citeseer')[0]
        self.dataset = Planetoid(root='./dataset/citeseer', name='citeseer')
        self.label =self.data.y
        self.data.y = onehot(self.data.y)
        self.num_classes = self.data.y.size()[1]
        self.y_train, self.y_test, self.y_val = self.data.y[self.data.train_mask], self.data.y[self.data.test_mask], self.data.y[self.data.val_mask]
        self.adj = gcn_norm(self.data.edge_index)
        data_augmentation(self,'citeseer')
        self.x_train, self.x_test, self.x_val = self.x[self.data.train_mask], self.x[self.data.test_mask], self.x[
            self.data.val_mask]
        self.train_mask, self.test_mask, self.val_mask = self.data.train_mask, self.data.test_mask, self.data.val_mask



class amazon_computers():
    def __init__(self):
        self.data = Amazon(root='./dataset/computers', name='computers')[0]
        self.dataset = Amazon(root='./dataset/computers', name='computers')
        self.label = self.data.y
        self.data.y = onehot(self.data.y)
        self.num_classes = self.data.y.size()[1]  # num_classes = 10
        self.adj = gcn_norm(self.data.edge_index)
        split_dataset(self)
        self.y_train, self.y_test, self.y_val = self.data.y[self.train_mask], self.data.y[self.test_mask], self.data.y[self.val_mask]
        data_augmentation(self,'amazon_computers')
        self.x_train, self.x_test, self.x_val = self.x[self.train_mask], self.x[self.test_mask], self.x[
            self.val_mask]



class amazon_photo():
    def __init__(self):
        self.data = Amazon(root='./dataset/photo', name='photo')[0]
        self.dataset = Amazon(root='./dataset/photo', name='photo')
        self.label = self.data.y
        self.data.y = onehot(self.data.y)
        self.num_classes = self.data.y.size()[1]
        self.adj = gcn_norm(self.data.edge_index)
        split_dataset(self)
        self.y_train, self.y_test, self.y_val = self.data.y[self.train_mask], self.data.y[self.test_mask], self.data.y[self.val_mask]
        data_augmentation(self,'amazon_photo')
        self.x_train, self.x_test, self.x_val = self.x[self.train_mask], self.x[self.test_mask], self.x[
            self.val_mask]


class coauthor_cs():
    def __init__(self):
        self.data = Coauthor(root='./dataset/cs', name='cs')[0]
        self.dataset = Coauthor(root='./dataset/cs', name='cs')
        self.label = self.data.y
        self.data.y = onehot(self.data.y)
        self.num_classes = self.data.y.size()[1]
        self.adj = gcn_norm(self.data.edge_index)
        split_dataset(self)
        self.y_train, self.y_test, self.y_val = self.data.y[self.train_mask], self.data.y[self.test_mask], self.data.y[self.val_mask]
        data_augmentation(self,'coauthor_cs')
        self.x_train, self.x_test, self.x_val = self.x[self.train_mask], self.x[self.test_mask], self.x[
            self.val_mask]



class ogbn_arxiv():
    def __init__(self):
        self.dataset = PygNodePropPredDataset(name='ogbn-arxiv',
                                              root='./dataset',
                                 transform=T.ToSparseTensor())
        self.processed_dir = self.dataset.processed_dir
        self.data = self.dataset[0]
        self.data.adj_t = self.data.adj_t.to_symmetric()

        row, col, val = self.data.adj_t.coo()
        self.edge_index = torch.stack([row, col], dim=0)
        self.adj = gcn_norm(self.edge_index)

        self.x = self.data.x
        self.label = self.data.y.squeeze()
        self.data.y = onehot(self.label)
        # self.label_onehot = onehot(self.label)
        self.num_classes = max(self.label) + 1
        self.num_features = self.data.x.size()[1]

        split_idx = self.dataset.get_idx_split()
        train_idx = split_idx['train']
        test_idx = split_idx['test']
        val_idx = split_idx['valid']
        self.train_mask = torch.zeros_like(self.label).bool().fill_(False)
        self.test_mask = torch.zeros_like(self.label).bool().fill_(False)
        self.val_mask = torch.zeros_like(self.label).bool().fill_(False)
        self.train_mask[train_idx] = True
        self.test_mask[test_idx] = True
        self.val_mask[val_idx] = True
        self.y_train, self.y_test, self.y_val = self.data.y[self.train_mask], self.data.y[self.test_mask], self.data.y[
            self.val_mask]
        data_augmentation(self,'ogbn_arxiv')
        self.x_train, self.x_test, self.x_val = self.x[self.train_mask], self.x[self.test_mask], self.x[
            self.val_mask]


class ogbn_products():
    def __init__(self):
        self.dataset = PygNodePropPredDataset(name='ogbn-products',
                                              root='./dataset',
                                 transform=T.ToSparseTensor())
        self.processed_dir = self.dataset.processed_dir
        self.data = self.dataset[0]
        self.data.adj_t = self.data.adj_t.to_symmetric()

        row, col, val = self.data.adj_t.coo()
        self.edge_index = torch.stack([row, col], dim=0)
        self.adj = gcn_norm(self.edge_index)

        self.x = self.data.x
        self.label = self.data.y.squeeze()
        self.data.y = onehot(self.label)
        # self.label_onehot = onehot(self.label)
        self.num_classes = max(self.label) + 1
        self.num_features = self.data.x.size()[1]

        split_idx = self.dataset.get_idx_split()
        train_idx = split_idx['train']
        test_idx = split_idx['test']
        val_idx = split_idx['valid']
        self.train_mask = torch.zeros_like(self.label).bool().fill_(False)
        self.test_mask = torch.zeros_like(self.label).bool().fill_(False)
        self.val_mask = torch.zeros_like(self.label).bool().fill_(False)
        self.train_mask[train_idx] = True
        self.test_mask[test_idx] = True
        self.val_mask[val_idx] = True
        self.y_train, self.y_test, self.y_val = self.data.y[self.train_mask], self.data.y[self.test_mask], self.data.y[
            self.val_mask]
        data_augmentation(self,'ogbn_products')
        self.x_train, self.x_test, self.x_val = self.x[self.train_mask], self.x[self.test_mask], self.x[
            self.val_mask]

class flickr():
    def __init__(self):
        self.data = Flickr(root='./dataset/Flickr')[0]
        self.processed_dir = Flickr(root='./dataset/Flickr').processed_dir
        self.x =self.data.x.to('cpu')
        self.label = self.data.y
        self.num_classes = max(self.label) + 1
        self.num_features = self.data.x.size()[1]
        # split the dataset
        # split_dataset(self)
        self.train_mask = self.data.train_mask
        self.test_mask = self.data.test_mask
        self.val_mask = self.data.val_mask
        # self.label_train, self.label_test = self.data.y[self.data.train_mask].to('cpu'), self.data.y[self.data.test_mask].to('cpu')
        self.data.y = onehot(self.label).to('cpu')
        self.adj = gcn_norm(self.data.edge_index)
        self.y_train, self.y_test, self.y_val = self.data.y[self.train_mask], self.data.y[self.test_mask], self.data.y[
            self.val_mask]
        data_augmentation(self,'flickr')
        self.x_train, self.x_test, self.x_val = self.x[self.train_mask], self.x[self.test_mask], self.x[
            self.val_mask]


class reddit2():
    def __init__(self):
        self.data = Reddit2(root='tmp/Reddit2')[0]
        self.processed_dir = Reddit2(root='tmp/Reddit2').processed_dir
        self.x =self.data.x.to('cpu')
        self.label = self.data.y.to('cpu')

        # self.label_onehot = onehot(self.data.y)
        self.num_classes = max(self.data.y)+1
        self.num_features = self.data.x.size()[1]
        self.train_mask = self.data.train_mask
        self.test_mask = self.data.test_mask
        self.val_mask = self.data.val_mask
        self.data.y = onehot(self.label).to('cpu')
        self.y_train, self.y_test, self.y_val = self.data.y[self.train_mask], self.data.y[self.test_mask], self.data.y[
            self.val_mask]
        self.adj = gcn_norm(self.data.edge_index).to('cpu')
        data_augmentation(self,'reddit2')
        self.x_train, self.x_test, self.x_val = self.x[self.train_mask], self.x[self.test_mask], self.x[
            self.val_mask]


class reddit():
    def __init__(self):
        self.data = Reddit(root='tmp/Reddit')[0]
        self.processed_dir = Reddit2(root='tmp/Reddit').processed_dir
        self.x =self.data.x.to('cpu')
        self.label = self.data.y.to('cpu')

        # self.label_onehot = onehot(self.data.y)
        self.num_classes = max(self.data.y)+1
        self.num_features = self.data.x.size()[1]
        self.train_mask = self.data.train_mask
        self.test_mask = self.data.test_mask
        self.val_mask = self.data.val_mask
        self.data.y = onehot(self.label).to('cpu')
        self.y_train, self.y_test, self.y_val = self.data.y[self.train_mask], self.data.y[self.test_mask], self.data.y[
            self.val_mask]
        self.adj = gcn_norm(self.data.edge_index).to('cpu')
        data_augmentation(self,'reddit')
        self.x_train, self.x_test, self.x_val = self.x[self.train_mask], self.x[self.test_mask], self.x[
            self.val_mask]



def split_dataset(data):
    num_train_per_class = 20
    num_test = 1000
    num_val = 1000
    data.train_mask = torch.zeros(size=(1, data.label.size()[0])).squeeze(dim=0).bool().fill_(False)
    data.test_mask = torch.zeros(size=(1, data.label.size()[0])).squeeze(dim=0).bool().fill_(False)
    data.val_mask = torch.zeros(size=(1, data.label.size()[0])).squeeze(dim=0).bool().fill_(False)
    # train mask
    for c in range(data.num_classes):
        idx = (data.label == c).nonzero().view(-1)
        torch.manual_seed(seed=100)
        idx = idx[torch.randperm(idx.size(0))[:num_train_per_class]]
        data.train_mask[idx] = True
    # test mask and val mask
    remaining = (~data.train_mask).nonzero().view(-1)
    torch.manual_seed(seed=100)
    remaining = remaining[torch.randperm(remaining.size(0))]
    data.test_mask[remaining[:num_test]] = True
    data.val_mask[remaining[num_test:num_test + num_val]] = True

def data_augmentation(data, name=None):
    # product1 = az(data.adj, data.data.x)
    # product2 = az(data.adj, product1)
    # product3 = az(data.adj, product2)
    # product4 = az(data.adj, product3)
    # data.x = torch.cat((product4, product3, product2, product1, data.data.x), dim=1)
    if name and os.path.exists(name+'_preprocess.pt'):
        f = torch.load(name+'_preprocess.pt')
        data.x = f['x']
        print('Loaded:' + name+'_preprocess.pt')

    else:
        product1 = az(data.adj, data.x)
        print('computed product1')
        # torch.save({'product1':product1}, name + 'product1.pt')
        product2 = az(data.adj, product1)
        print('computed product2')
        # torch.save({'product2': product2}, name + 'product2.pt')
        product3 = az(data.adj, product2)
        print('computed product3')
        # torch.save({'product3': product3}, name + 'product3.pt')
        product4 = az(data.adj, product3)
        print('computed product4')
        # torch.save({'product4': product4}, name + 'product4.pt')
        data.x = torch.cat((product4, product3, product2, product1, data.data.x), dim=1)
        torch.save({'x': data.x}, name + '_preprocess.pt')
        print('saved: {}'.format(name + '_preprocess.pt'))
