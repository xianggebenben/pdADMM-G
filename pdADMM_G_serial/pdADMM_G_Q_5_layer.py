import torch
import numpy as np
import sys
import time

if not sys.warnoptions:
    import warnings

    warnings.simplefilter("ignore")
import GA_MLP.common as common
from GA_MLP.input_data import cora, pubmed, citeseer, amazon_computers, amazon_photo, coauthor_cs,coauthor_physics, ogbn_arxiv, flickr

import torch.nn.functional as F


# initialize the neural network
def Net(graph, num_of_neurons, seed_num):
    data = torch.load("pdadmm_g_q_"+dataset_name+"_"+str(num_of_neurons)+"_"+repr(seed_num)+"_2layers.pt")
    W1 = data['W1'].to(device)
    b1 = data['b1'].to(device)
    z1 = data['z1'].to(device)
    q1 = data['q1'].to(device)
    del data
    p2 = q1
    W2 = torch.eye(num_of_neurons).to(device)
    b2 = torch.zeros(size=(num_of_neurons, 1)).to(device)
    z2 = torch.matmul(W2, p2) + b2
    q2 = F.relu(z2)
    p3 = q2
    W3 = torch.eye(num_of_neurons).to(device)
    b3 = torch.zeros(size=(num_of_neurons, 1)).to(device)
    z3 = torch.matmul(W3, p3) + b3
    q3 = F.relu(z3)
    p4 = q3
    W4 = torch.eye(num_of_neurons).to(device)
    b4 = torch.zeros(size=(num_of_neurons, 1)).to(device)
    z4 = torch.matmul(W4, p4) + b4
    q4 = F.relu(z4)
    p5 = q4
    torch.manual_seed(seed=seed_num)
    W5 = torch.normal(size=(num_classes, num_of_neurons), mean=0, std=0.1, requires_grad=True).to(device)
    torch.manual_seed(seed=seed_num)
    b5 = torch.normal(size=(num_classes, 1), mean=0, std=0.1, requires_grad=True).to(device)
    z5 = torch.zeros(size=(num_classes, graph.size()[1]), requires_grad=True).to(device)

    # z5.required_grad=True
    return W1, b1, z1, q1, p2, W2, b2, z2, q2, p3, W3, b3, z3, q3, p4, W4, b4, z4, q4, p5, W5, b5, z5


# return the accuracy of the neural network model
def test_accuracy(W1, b1, W2, b2, W3, b3, W4, b4, W5, b5, graph, labels):
    nums = int(labels.shape[1])
    z1 = torch.matmul(W1, graph) + b1
    q1 = F.relu(z1)
    p2 = q1
    z2 = torch.matmul(W2, p2) + b2
    q2 = F.relu(z2)
    p3 = q2
    z3 = torch.matmul(W3, p3) + b3
    q3 = F.relu(z3)
    p4 = q3
    z4 = torch.matmul(W4, p4) + b4
    q4 = F.relu(z4)
    p5 = q4
    z5 = torch.matmul(W5, p5) + b5
    cost = common.cross_entropy_with_softmax(labels, z5) / nums
    label = torch.argmax(labels, dim=0)
    pred = torch.argmax(z5, dim=0)
    return (torch.sum(torch.eq(pred, label), dtype=torch.float32).item() / nums, cost)

def objective(graph, labels,
              W1, b1, z1, q1, u1,
              p2, W2, b2, z2, q2,u2,
              p3, W3, b3, z3, q3,u3,
              p4, W4, b4, z4, q4,u4,
              p5, W5, b5, z5,rho,mu):
    p1 = graph
    loss = common.cross_entropy_with_softmax(labels, z5)
    penalty = 0
    # res = 0
    for j in range(1, 6):
        temp1 = locals()['z' + str(j)] - locals()['W' + str(j)].matmul(locals()['p' + str(j)]) - locals()['b' + str(j)]
        temp4 = locals()['W'+str(j)]
        if j<=4:
            temp2 = locals()['q' + str(j)] - F.relu(locals()['z' + str(j)])
            temp3 = locals()['p' + str(j + 1)] - locals()['q' + str(j)]
            penalty += rho / 2 * torch.sum(temp1 * temp1) + rho / 2 * torch.sum(temp2 * temp2) \
                       + torch.sum((rho / 2 * temp3 + locals()['u' + str(j)]) * temp3) + mu / 2 * torch.sum(
                temp4 * temp4)
        else:
            penalty += rho / 2 * torch.sum(temp1 * temp1) + mu / 2 * torch.sum(
                temp4 * temp4)

    obj = loss + penalty
    return obj

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")
dataset_name = 'cora'
# cora, pubmed, citeseer, amazon_computers, amazon_photo, coauthor_cs, ogbn_arxiv, flickr
print('dataset: {}'.format(dataset_name))
if dataset_name == 'cora':
    dataset = cora()
    rho = 1e-5
    mu = 0.01
elif dataset_name == 'pubmed':
    dataset = pubmed()
    rho = 1e-5
    mu = 0.1
elif dataset_name == 'citeseer':
    dataset = citeseer()
    rho = 1e-5
    mu = 0.1
elif dataset_name == 'amazon_computers':
    dataset = amazon_computers()
    rho = 1e-3
    mu = 1
elif dataset_name == 'amazon_photo':
    dataset = amazon_photo()
    rho = 1e-3
    mu = 1
elif dataset_name == 'coauthor_cs':
    dataset = coauthor_cs()
    rho = 1e-5
    mu = 0.1
elif dataset_name == 'coauthor_physics':
    dataset = coauthor_physics()
    rho = 1e-5
    mu = 0.1
elif dataset_name == 'ogbn_arxiv':
    dataset = ogbn_arxiv()
    rho = 1e-5
    mu = 0.1
elif dataset_name == 'flickr':
    dataset = flickr()
    rho = 1e-5
    mu = 0.1
else:
    raise ValueError('Please type the correct dataset name.')


# initialization
x_train = dataset.x_train.to(device)
x_train = torch.transpose(x_train, 0, 1)
y_train = dataset.y_train.to(device)
y_train = torch.transpose(y_train, 0, 1)
x_test = dataset.x_test.to(device)
x_test = torch.transpose(x_test, 0, 1)
y_test = dataset.y_test.to(device)
y_test = torch.transpose(y_test, 0, 1)
x_val = dataset.x_val.to(device)
x_val = torch.transpose(x_val, 0, 1)
y_val = dataset.y_val.to(device)
y_val = torch.transpose(y_val, 0, 1)
num_classes = dataset.num_classes
del dataset
num_of_neurons = 500
ITER = 200
index = 0
seed_nums = [0,100,200,300,400]
delta = np.linspace(-1, 2, 1000)
delta = torch.from_numpy(delta).unsqueeze(dim=-1).float().to(device)
rho_back=rho
for seed_num in seed_nums:
    rho = rho_back
    t_train = 0
    print('-' * 20)
    print('seed={}'.format(seed_num))
    W1, b1, z1, q1, p2, W2, b2, z2, q2, p3, W3, b3, z3, q3, p4, W4, b4, z4, q4, p5, W5, b5, z5 = Net(x_train,
                                                                                                     num_of_neurons, seed_num)


    u1 = torch.zeros(q1.shape).to(device)
    u2 = torch.zeros(q2.shape).to(device)
    u3 = torch.zeros(q3.shape).to(device)
    u4 = torch.zeros(q4.shape).to(device)
    train_acc = np.zeros(ITER)
    val_acc = np.zeros(ITER)
    train_cost = np.zeros(ITER)
    val_cost = np.zeros(ITER)

    for i in range(ITER):
        pre = time.time()
        print('-'*20)
        print("iter=", i)
        # p2 = common.update_p_quantize(p2, q1, W2, b2, z2, u1, rho,delta)
        # p3 = common.update_p_quantize(p3, q2, W3, b3, z3, u2, rho,delta)
        # p4 = common.update_p_quantize(p4, q3, W4, b4, z4, u3, rho,delta)
        # p5 = common.update_p_quantize(p5, q4, W5, b5, z5, u4, rho,delta)

        p2 = common.update_p(p2, q1, W2, b2, z2, u1, rho)
        p3 = common.update_p(p3, q2, W3, b3, z3, u2, rho)
        p4 = common.update_p(p4, q3, W4, b4, z4, u3, rho)
        p5 = common.update_p(p5, q4, W5, b5, z5, u4, rho)

        W1 = common.update_W(x_train, b1, z1, W1, rho, mu)
        W2 = common.update_W(p2, b2, z2, W2, rho, mu)
        W3 = common.update_W(p3, b3, z3, W3, rho, mu)
        W4 = common.update_W(p4, b4, z4, W4, rho, mu)
        W5 = common.update_W(p5, b5, z5, W5, rho, mu)
        b1 = common.update_b(x_train, W1, z1, b1, rho)
        b2 = common.update_b(p2, W2, z2, b2, rho)
        b3 = common.update_b(p3, W3, z3, b3, rho)
        b4 = common.update_b(p4, W4, z4, b4, rho)
        b5 = common.update_b(p5, W5, z5, b5, rho)
        z1 = common.update_z(z1, x_train, W1, b1, q1, rho)
        z2 = common.update_z(z2, p2, W2, b2, q2, rho)
        z3 = common.update_z(z3, p3, W3, b3, q3, rho)
        z4 = common.update_z(z4, p4, W4, b4, q4, rho)
        z5 = common.update_zl(p5, W5, b5, y_train, z5, rho)

        # q1 = common.update_q(p2, z1, u1, rho)
        # q2 = common.update_q(p3, z2, u2, rho)
        #common.plot_histogram(q1)
        # q3 = common.update_q(p4, z3, u3, rho)
        # q4 = common.update_q(p5, z4, u4, rho)

        q1 = common.update_q_quantize(p2, z1, u1, rho,delta)
        q2 = common.update_q_quantize(p3, z2, u2, rho,delta)
        q3 = common.update_q_quantize(p4, z3, u3, rho,delta)
        q4 = common.update_q_quantize(p5, z4, u4, rho,delta)
        r1 = p2-q1
        r2 = p3-q2
        r3 = p4-q3
        r4 = p5-q4
        u1 = u1 + rho * r1
        u2 = u2 + rho * r2
        u3 = u3 + rho * r3
        u4 = u4 + rho * r4
        t_train += time.time() - pre
        print("Time per iteration:", time.time() - pre)
        print("rho=", rho)
        (train_acc[i], train_cost[i]) = test_accuracy(W1, b1, W2, b2, W3, b3, W4, b4, W5, b5, x_train, y_train)
        print("training cost:", train_cost[i])
        print("training acc:", train_acc[i])
        (val_acc[i], val_cost[i]) = test_accuracy(W1, b1, W2, b2, W3, b3, W4, b4, W5, b5, x_val, y_val)
        print("validation cost:", val_cost[i])
        print("validation acc:", val_acc[i])
        obj=objective(x_train, y_train, W1, b1, z1, q1, u1,
                  p2, W2, b2, z2, q2,u2,
                  p3, W3, b3, z3, q3,u3,
                  p4, W4, b4, z4, q4,u4,
                  p5, W5, b5, z5,rho,mu)
        linear_r=torch.sum(r1*r1+r2*r2+r3*r3+r4*r4)
        print("obj:",obj.cpu().detach().numpy())
        print("linear_r:", linear_r.cpu().detach().numpy())
        # if i > 2 and train_cost[i] > train_cost[i - 1]-0.001 \
        #     and train_cost[i - 1] > train_cost[i - 2]-0.001 :

        # cora
        if i > 2 and train_cost[i] > train_cost[i - 1]-0.01 \
            and train_cost[i - 1] > train_cost[i - 2]-0.01 :
               rho = np.minimum(10 * rho, 1)

    (test_acc, test_cost) = test_accuracy(W1, b1, W2, b2, W3, b3, W4, b4, W5, b5, x_test, y_test)
    print("test cost:", test_cost.item())
    print("test acc:", test_acc)
    t_train /= ITER
    torch.save(
        {"W1": W1, "b1": b1, "z1": z1, "q1": q1, "u1": u1,
         "p2": p2, "W2": W2, "b2": b2, "z2": z2, "q2": q2, "u2": u2,
         "p3": p3, "W3": W3, "b3": b3, "z3": z3, "q3": q3, "u3": u3,
         "p4": p4, "W4": W4, "b4": b4, "z4": z4, "q4": q4, "u4": u4,
         "p5": p5, "W5": W5, "b5": b5, "z5": z5,
         "rho": rho}, \
        './pdadmm_g_q_'+dataset_name+'_' + repr(num_of_neurons) +'_'+repr(seed_num)+ '_5layers.pt')
    torch.save(
        {"linear_r": linear_r, "obj": obj,
         "train_acc": train_acc, "train_cost": train_cost, "val_acc": val_acc, "val_cost": val_cost,
         "test_acc": test_acc, "test_cost": test_cost, 't_train': t_train},
        './pdadmm_g_q_' + dataset_name + '_' + repr(num_of_neurons) + '_' + repr(seed_num) + '_5layers_acc.pt')
