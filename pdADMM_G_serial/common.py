import torch
import numpy as np
import torch.nn.functional as F
from torch_sparse import SparseTensor, fill_diag, sum, mul
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_geometric.utils import add_remaining_self_loops
from torch.autograd import Variable
from torch_scatter import scatter_add
import matplotlib.pyplot as plt
import torch.nn.init as init
import math
#device = torch.device("cpu")
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def plot_histogram(p):
    # vectorize the matrix row-wise
    p_vec = p.reshape(-1).unsqueeze(dim=-1)
    p_vec = p_vec.detach().numpy()
    num_bins = 100  # quantize intervals
    fig, ax = plt.subplots()
    # the histogram of the data
    n, bins, patches = ax.hist(p_vec, num_bins, density=1)
    ax.set_xlabel('p')
    ax.set_ylabel('Probability density')
    ax.set_title(r'Histogram of p')

    # Tweak spacing to prevent clipping of ylabel
    fig.tight_layout()
    plt.ion()
    plt.pause(0.5)
    plt.close()

def reset_parameters(weight, bias):
    init.kaiming_uniform_(weight, a=math.sqrt(5))

    fan_in, _ = init._calculate_fan_in_and_fan_out(weighdt)
    bound = 1 / math.sqrt(fan_in)
    init.uniform_(bias, -bound, bound)
    return weight, bias

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
            edge_weight = torch.ones((edge_index.size(1), ), dtype=dtype,
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
        n = torch.max(edge_index[0])+1  # num of nodes
        adj_norm = SparseTensor(row=edge_index[0], col=edge_index[1], value=edge_weight,
                                  sparse_sizes=(n, n), is_sorted=True)
        return adj_norm
def cross_entropy_with_softmax(label, zl):
    prob = softmax(zl)
    loss = cross_entropy(label, prob)
    return loss
def softmax(x):
    exp =torch.exp(x)
    imask = torch.eq(exp, float("inf"))
    exp = torch.where(imask, torch.exp(torch.tensor(88.6).to(x.device))*torch.ones(size=exp.size()).to(x.device), exp)
    return exp/(torch.sum(exp,dim=0)+1e-10)
def cross_entropy(label, prob):
    loss = -torch.sum(label * torch.log(prob+1e-10))
    return loss
#return the  F.relu function
# return phi
def eq1(p, W, b, z, rho, mu):
    temp = z - torch.matmul(W, p) - b
    # res = rho / 2 * torch.sum(temp * temp)+mu/2*torch.sum(W*W)
    res = rho / 2 * torch.sum(temp * temp)+mu/2*torch.sum(W * W)
    return res
# # return the derivative of phi with regard to W
# def eq1_W(p, W, b, z,rho, mu):
#     temp = torch.matmul(W, p) + b - z
#     temp2 = torch.transpose(p,0,1)
#     res = rho * torch.matmul(temp, temp2)+mu*W
#     return res
# return the derivative of phi with regard to b
def eq1_b(p, W, b, z, rho):
    res = torch.reshape(torch.mean(rho * (torch.matmul(W, p) + b - z), dim=1),shape=(-1, 1))
    return res
# # return the derivative of phi with regard to z
# def eq1_z(a, W, b, z, rho):
#     res = rho * (z - b - torch.matmul(W, a))
#     return res
# return the quadratic approximation of W-subproblem
# def P(W_new, theta, p, W_old, b, z, rho, mu):
def P(W_new, theta, W_old, gradient, f):
    temp = W_new - W_old
    res = f + torch.sum((theta/2 * temp + gradient) * temp)
    # res =  eq1(p, W_old, b, z, rho, mu) \
    #  + torch.sum(eq1_W(p, W_old, b, z,rho,mu) * temp) \
    #       + torch.sum(theta * temp * temp) / 2
    return res
# return the quadratic approximation of p-subproblem
# def Q(p_new, tau, p_old, q, W, b, z, u, gradient, rho):
def Q(p_new, tau, p_old, gradient, f):
    temp = p_new - p_old
    # res = p_obj(p_old, q, W, b, z, u, rho) + torch.sum(gradient * temp) + torch.sum(
    #     tau * temp * temp) / 2
    res = f + torch.sum((tau/2 * temp + gradient) * temp)
    return res
def p_obj(p,q,W,b,z,u,rho):
    # f =rho/2*torch.sum((z-torch.matmul(W,p)-b)*(z-torch.matmul(W,p)-b))\
    f =rho/2*torch.sum((z-torch.matmul(W,p)-b)**2)+torch.sum((u+rho/2 * (p-q)) * (p-q))
       # +torch.sum(u * (p-q))+rho/2*torch.sum((p-q)*(p-q))
    return f
# # return the derivative of p_obj wrt p
# def p_obj_p(p,q,W,b,z,u,rho):
#     res =rho*torch.transpose(W,0,1).matmul(torch.matmul(W,p)+b-z)+u+rho*(p-q)
#     return res
def update_p(p_old,q,W, b, z,u,rho):
    p_old = Variable(p_old, requires_grad=True)
    f = p_obj(p_old, q,W, b, z, u,rho)
    gradient = torch.autograd.grad(f, inputs=[p_old])[0]
    # gradient =p_obj_p(p_old,q,W,b,z,u,rho)
    eta = 2
    t=20
    beta=p_old-gradient/t
    count=0
    # while (p_obj(beta, q,W, b, z, u,rho) > Q(beta, t_train, p_old, q, W, b, z, u,gradient,rho)):
    while (p_obj(beta, q,W, b, z, u,rho) > Q(beta, t, p_old, gradient,f)):
        t = t * eta
        beta=p_old-gradient/t
        count+=1
        if count>10:
            beta =p_old
            break
    tau = t
    p = beta
    return p
# return the result of W-subproblem
def update_W(p, b, z, W_old, rho, mu):
    W_old = Variable(W_old, requires_grad=True)
    f = eq1(p, W_old, b, z, rho, mu)
    gradient = torch.autograd.grad(f, inputs=[W_old])[0]
    # gradient = eq1_W(p, W_old, b, z, rho, mu)
    gamma = 2
    alpha = 20
    zeta = W_old - gradient / alpha
    count=0
    # while (eq1(p, zeta, b, z, rho, mu) > P(zeta, alpha, p, W_old, b, z,rho, mu)):
    while (eq1(p, zeta, b, z, rho, mu) > P(zeta, alpha, W_old, gradient, f)):
        alpha = alpha * gamma
        zeta = W_old - gradient / alpha  # Learning rate decreases to 0, leading to infinity loop here.
        count += 1
        if count > 10:
            zeta = W_old
            break
    theta = alpha
    W = zeta
    return W
def update_p_quantize(p_old,q,W, b, z,u,rho, delta):
    p_old = Variable(p_old, requires_grad=True)
    f = p_obj(p_old, q, W, b, z, u, rho)
    gradient = torch.autograd.grad(f, inputs=[p_old])[0]
    # gradient = p_obj_p(p_old,q, W, b, z, u, rho)
    eta = 2
    t = 20
    beta = p_old - gradient / t
    p = beta
    # quantization
    p_vec = p.reshape(-1).unsqueeze(dim=0)
    p_repeat = p_vec.repeat(delta.size()[0], 1)
    delta_repeat = delta.repeat(1, p_vec.size()[1])
    res = abs(p_repeat - delta_repeat)
    idx = torch.argmin(res, dim=0)
    p_quantized_vec = delta[idx]
    p_quantized = p_quantized_vec.reshape(p.size()[0], p.size()[1])
    count = 0
    # while (p_obj(p_quantized, q, W, b, z, u, rho) > Q(p_quantized, t_train, p_old, q, W, b, z, u, gradient, rho)):
    while (p_obj(p_quantized, q, W, b, z, u, rho) > Q(p_quantized, t, p_old, gradient, f)):
        t = t * eta
        beta = p_old - gradient / t
        p = beta
        # quantization
        p_vec = p.reshape(-1).unsqueeze(dim=0)
        p_repeat = p_vec.repeat(delta.size()[0], 1)
        delta_repeat = delta.repeat(1, p_vec.size()[1])
        res = abs(p_repeat - delta_repeat)
        idx = torch.argmin(res, dim=0)
        p_quantized_vec = delta[idx]
        p_quantized = p_quantized_vec.reshape(p.size()[0], p.size()[1])
        count += 1
        if count > 10:
            beta = p_old
            p = beta
            # quantization
            p_vec = p.reshape(-1).unsqueeze(dim=0)
            p_repeat = p_vec.repeat(delta.size()[0], 1)
            delta_repeat = delta.repeat(1, p_vec.size()[1])
            res = abs(p_repeat - delta_repeat)
            idx = torch.argmin(res, dim=0)
            p_quantized_vec = delta[idx]
            p_quantized = p_quantized_vec.reshape(p.size()[0], p.size()[1])
            break
    tau = t
    # plot_histogram(p)
    return p_quantized
# return the result of b-subproblem
def update_b(p, W, z, b_old, rho):
    gradients = eq1_b(p, W, b_old, z, rho)
    res = b_old - gradients / rho
    return res
# return the objective value of z-subproblem
def z_obj(p, W, b, z, q,rho):
    # f=rho/2*(z-torch.matmul(W,p)-b)*(z-torch.matmul(W,p)-b)+rho/2*(q-F.relu(z))*(q-F.relu(z))
    f=rho/2*(z-torch.matmul(W,p)-b) ** 2 + rho/2*(q-F.relu(z)) ** 2
    return f
# return the result of z-subproblem
def update_z(z,p, W, b, q, rho):
    z1=(torch.matmul(W,p)+b+z)/2
    z2=(2*z1+q)/3
    z1=torch.min(z1,torch.zeros(size=z1.size()).to(z1.device))
    z2=torch.max(z2,torch.zeros(size=z2.size()).to(z1.device))
    value1=z_obj(p, W, b, z1, q,rho)
    value2=z_obj(p, W, b, z2, q,rho)
    imask =torch.gt(value1, value2)
    z=torch.where(imask, z2,z1)
    return z
# return the result of z_L-subproblem by FISTA
def update_zl(p, W, b, label, zl_old, rho):
    fzl = 10e10
    MAX_ITER = 50
    zl = zl_old
    lamda = 1
    zeta = zl
    eta = 4
    TOLERANCE = 1e-3
    for i in range(MAX_ITER):
        fzl_old = fzl
        # fzl = cross_entropy_with_softmax(label, zl)+rho/2*torch.sum((zl-torch.matmul(W,p)-b)*(zl-torch.matmul(W,p)-b))
        fzl = cross_entropy_with_softmax(label, zl)+rho/2*torch.sum((zl-torch.matmul(W,p)-b)**2)
        if abs(fzl - fzl_old) < TOLERANCE:
            break
        lamda_old = lamda
        lamda = (1 + np.sqrt(1 + 4 * lamda * lamda)) / 2
        gamma = (1 - lamda_old) / lamda
        gradients2 = (softmax(zl) - label)
        zeta_old = zeta
        zeta = (rho * (torch.matmul(W, p)+b) + (zl - eta * gradients2) / eta) / (rho + 1 / eta)
        zl = (1 - gamma) * zeta + gamma * zeta_old
    return zl
def update_q(p,z,u,rho):
    res =(p+u/rho+F.relu(z))/2
    return res
def update_q_quantize(p,z,u,rho,delta):
    device = p.device
    delta = delta.cpu()
    q =((p+u/rho+F.relu(z))/2).cpu()
    q = q.reshape(-1).unsqueeze(dim=0).cpu()
    q_repeat = q.repeat(delta.size()[0], 1).cpu()
    delta_repeat = delta.repeat(1, q.size()[1]).cpu()
    res = abs(q_repeat - delta_repeat)
    idx = torch.argmin(res, dim=0)
    q = delta[idx]
    q = q.reshape(p.size()[0], p.size()[1]).to(device)
    return q