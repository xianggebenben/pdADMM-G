from torch_geometric.datasets import Planetoid, Amazon, Coauthor
import numpy as np
import torch
from torch_sparse import spmm
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_sparse import SparseTensor, fill_diag, sum, mul
from torch_geometric.utils import add_remaining_self_loops,add_self_loops
from torch_scatter import scatter_add


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


def azw(adj, z):
    if isinstance(adj, SparseTensor):
        # if adj.is_sparse:
        #     return torch.sparse.mm(adj, z).matmul(w)
        #adj =adj+torch.sparse_coo_tensor(torch.eye(adj.size(dim=0).item()))
        row, col, adj_value = adj.coo()
        edge_index = torch.stack([row, col], dim=0)
        edge_index,adj_value =add_self_loops(edge_index,adj_value,1,adj.size(0))
        # A*Z then (A*Z)*W
        # # pre_spmm = time.time()
        # temp = spmm(edge_index, adj_value, z.size()[0], z.size()[0], z)
        # # print('time for spmm:', time.time() - pre_spmm)
        # return temp.matmul(w)
        # usage: https://github.com/rusty1s/pytorch_sparse

        # Z*W THEN A*(A*W)  much more efficient!!
        return spmm(edge_index, adj_value, z.size()[0], z.size()[0], z)
    else:
        return adj.matmul(z)



class cora():
    def __init__(self):
        try:
            self.cora = Planetoid(root='/home/xd/Documents/code/admm_gnn/dataset/cora', name='cora')[0]

        except:
            self.cora = Planetoid(root='/dataset/cora', name='cora')[0]
            self.dataset = Planetoid(root='/dataset/cora', name='cora')[0]
        # self.cora = Planetoid(root='~/Documents/code/admm_gnn/dataset/cora', name='cora')[0]
        self.label = self.cora.y
        self.cora.y = onehot(self.cora.y)
        self.num_classes = self.cora.y.size()[1]
        self.y_train, self.y_test = self.cora.y[self.cora.train_mask], self.cora.y[self.cora.test_mask]
        self.adj = gcn_norm(self.cora.edge_index)
        self.product1=azw(self.adj, self.cora.x)
        self.product2=azw(self.adj, self.product1)
        self.product3=azw(self.adj, self.product2)
        self.product4=azw(self.adj, self.product3)
        self.x = torch.cat((self.product4,self.product3,self.product2,self.product1,self.cora.x),dim=1)
        self.x_train, self.x_test = self.x[self.cora.train_mask], self.x[self.cora.test_mask]
        self.train_mask, self.test_mask =self.cora.train_mask, self.cora.test_mask


class pubmed():
    def __init__(self):
        try:
            self.cora = Planetoid(root='/home/xd/Documents/code/admm_gnn/dataset/PubMed', name='PubMed')[0]

        except:
            self.cora = Planetoid(root='/dataset/PubMed', name='PubMed')[0]
            self.dataset = Planetoid(root='/dataset/PubMed', name='PubMed')
        # self.cora = Planetoid(root='/tmp/PubMed', name='PubMed')[0]
        self.label =self.cora.y
        self.cora.y = onehot(self.cora.y)
        self.num_classes = self.cora.y.size()[1]
        self.y_train, self.y_test = self.cora.y[self.cora.train_mask], self.cora.y[self.cora.test_mask]
        self.adj = gcn_norm(self.cora.edge_index)
        self.product1=azw(self.adj, self.cora.x)
        self.product2=azw(self.adj, self.product1)
        self.product3=azw(self.adj, self.product2)
        self.product4=azw(self.adj, self.product3)
        self.x = torch.cat((self.product4,self.product3,self.product2,self.product1,self.cora.x),dim=1)
        self.x_train, self.x_test = self.x[self.cora.train_mask], self.x[self.cora.test_mask]
        self.train_mask, self.test_mask =self.cora.train_mask, self.cora.test_mask


class citeseer():
    def __init__(self):
        try:
            self.cora = Planetoid(root='/home/xd/Documents/code/admm_gnn/dataset/citeseer', name='citeseer')[0]

        except:
            self.cora = Planetoid(root='/dataset/citeseer', name='citeseer')[0]
            self.dataset = Planetoid(root='/dataset/citeseer', name='citeseer')
        # self.cora = Planetoid(root='/tmp/citeseer', name='citeseer')[0]
        self.label =self.cora.y
        self.cora.y = onehot(self.cora.y)
        self.num_classes = self.cora.y.size()[1]
        self.y_train, self.y_test = self.cora.y[self.cora.train_mask], self.cora.y[self.cora.test_mask]
        self.adj = gcn_norm(self.cora.edge_index)
        self.product1=azw(self.adj, self.cora.x)
        self.product2=azw(self.adj, self.product1)
        self.product3=azw(self.adj, self.product2)
        self.product4=azw(self.adj, self.product3)
        self.x = torch.cat((self.product4,self.product3,self.product2,self.product1,self.cora.x),dim=1)
        self.x_train, self.x_test = self.x[self.cora.train_mask], self.x[self.cora.test_mask]
        self.train_mask, self.test_mask =self.cora.train_mask, self.cora.test_mask



class amazon_computers():
    def __init__(self):
        try:
            self.cora = Amazon(root='/home/xd/Documents/code/admm_gnn/dataset/computers', name='Computers')[0]

        except:
            self.cora = Amazon(root='/dataset/computers', name='computers')[0]
            self.dataset = Amazon(root='/dataset/computers', name='computers')
        # self.cora = Amazon(root='/tmp/computers', name='computers')[0]
        self.label = self.cora.y
        self.cora.y = onehot(self.cora.y)
        self.num_classes = self.cora.y.size()[1]  # num_classes = 10

        # split the dataset
        num_train_per_class = 20
        num_test = 1000
        self.train_mask = torch.zeros(size=(1, self.label.size()[0])).squeeze(dim=0).bool().fill_(False)
        self.test_mask = torch.zeros(size=(1, self.label.size()[0])).squeeze(dim=0).bool().fill_(False)

        for c in range(self.num_classes):
            idx = (self.label == c).nonzero().view(-1)
            torch.manual_seed(seed=100)
            idx = idx[torch.randperm(idx.size(0))[:num_train_per_class]]
            self.train_mask[idx] = True

        remaining = (~self.train_mask).nonzero().view(-1)
        torch.manual_seed(seed=100)
        remaining = remaining[torch.randperm(remaining.size(0))]
        # self.test_mask[remaining] = True
        self.test_mask[remaining[:num_test]] = True

        self.y_train, self.y_test = self.cora.y[self.train_mask], self.cora.y[self.test_mask]
        self.adj = gcn_norm(self.cora.edge_index)
        self.product1 = azw(self.adj, self.cora.x)
        self.product2 = azw(self.adj, self.product1)
        self.product3 = azw(self.adj, self.product2)
        self.product4 = azw(self.adj, self.product3)
        self.x = torch.cat((self.product4, self.product3, self.product2, self.product1, self.cora.x), dim=1)
        self.x_train, self.x_test = self.x[self.train_mask], self.x[self.test_mask]


class amazon_photo():
    def __init__(self):
        try:
            self.cora = Amazon(root='/home/xd/Documents/code/admm_gnn/dataset/photo', name='photo')[0]

        except:
            self.cora = Amazon(root='/dataset/photo', name='photo')[0]
            self.dataset = Amazon(root='/dataset/photo', name='photo')
        # self.cora = Amazon(root='/tmp/photo', name='photo')[0]
        self.label = self.cora.y
        self.cora.y = onehot(self.cora.y)
        self.num_classes = self.cora.y.size()[1]

        # split the dataset
        num_train_per_class = 20
        num_test = 1000
        self.train_mask = torch.zeros(size=(1, self.label.size()[0])).squeeze(dim=0).bool().fill_(False)
        self.test_mask = torch.zeros(size=(1, self.label.size()[0])).squeeze(dim=0).bool().fill_(False)

        for c in range(self.num_classes):
            idx = (self.label == c).nonzero().view(-1)
            torch.manual_seed(seed=100)
            idx = idx[torch.randperm(idx.size(0))[:num_train_per_class]]
            self.train_mask[idx] = True

        remaining = (~self.train_mask).nonzero().view(-1)
        torch.manual_seed(seed=100)
        remaining = remaining[torch.randperm(remaining.size(0))]
        self.test_mask[remaining[:num_test]] = True

        self.y_train, self.y_test = self.cora.y[self.train_mask], self.cora.y[self.test_mask]
        self.adj = gcn_norm(self.cora.edge_index)
        self.product1 = azw(self.adj, self.cora.x)
        self.product2 = azw(self.adj, self.product1)
        self.product3 = azw(self.adj, self.product2)
        self.product4 = azw(self.adj, self.product3)
        self.x = torch.cat((self.product4, self.product3, self.product2, self.product1, self.cora.x), dim=1)
        self.x_train, self.x_test = self.x[self.train_mask], self.x[self.test_mask]


class coauthor_cs():
    def __init__(self):
        try:
            self.cora = Coauthor(root='/home/xd/Documents/code/admm_gnn/dataset/cs', name='cs')[0]

        except:
            self.cora = Coauthor(root='/dataset/cs', name='cs')[0]
            self.dataset = Coauthor(root='/dataset/cs', name='cs')
        # self.cora = Coauthor(root='/tmp/cs', name='cs')[0]
        self.label = self.cora.y
        self.cora.y = onehot(self.cora.y)
        self.num_classes = self.cora.y.size()[1]

        # split the dataset
        num_train_per_class = 20
        num_test = 1000
        self.train_mask = torch.zeros(size=(1, self.label.size()[0])).squeeze(dim=0).bool().fill_(False)
        self.test_mask = torch.zeros(size=(1, self.label.size()[0])).squeeze(dim=0).bool().fill_(False)

        for c in range(self.num_classes):
            idx = (self.label == c).nonzero().view(-1)
            torch.manual_seed(seed=100)
            idx = idx[torch.randperm(idx.size(0))[:num_train_per_class]]
            self.train_mask[idx] = True

        remaining = (~self.train_mask).nonzero().view(-1)
        torch.manual_seed(seed=100)
        remaining = remaining[torch.randperm(remaining.size(0))]
        self.test_mask[remaining[:num_test]] = True

        self.y_train, self.y_test = self.cora.y[self.train_mask], self.cora.y[self.test_mask]
        self.adj = gcn_norm(self.cora.edge_index)
        self.product1 = azw(self.adj, self.cora.x)
        self.product2 = azw(self.adj, self.product1)
        self.product3 = azw(self.adj, self.product2)
        self.product4 = azw(self.adj, self.product3)
        self.x = torch.cat((self.product4, self.product3, self.product2, self.product1, self.cora.x), dim=1)
        self.x_train, self.x_test = self.x[self.train_mask], self.x[self.test_mask]