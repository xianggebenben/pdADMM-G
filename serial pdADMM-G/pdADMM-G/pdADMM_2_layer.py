import torch
import numpy as np
import sys
import time

if not sys.warnoptions:
    import warnings

    warnings.simplefilter("ignore")
from serial import common
from serial.input_data import cora, pubmed, citeseer, amazon_computers, amazon_photo, coauthor_cs
import torch.nn.functional as F


# initialize the neural network
def Net(graph, num_of_neurons):
    seed_num = 0
    torch.manual_seed(seed=seed_num)
    W1 = torch.normal(size=(num_of_neurons, graph.size()[0]), mean=0, std=0.1, requires_grad=True)
    torch.manual_seed(seed=seed_num)
    b1 = torch.normal(size=(num_of_neurons, 1), mean=0, std=0.1, requires_grad=True)
    z1 = torch.matmul(W1, graph) + b1
    q1 = F.relu(z1)
    p2 = q1
    W2 = torch.normal(size=(dataset.num_classes, num_of_neurons), mean=0, std=0.1, requires_grad=True)
    torch.manual_seed(seed=seed_num)
    b2 = torch.normal(size=(dataset.num_classes, 1), mean=0, std=0.1, requires_grad=True)
    z2 = torch.zeros(size=(dataset.num_classes, graph.size()[1]), requires_grad=True)

    # z5.required_grad=True
    return W1, b1, z1, q1, p2, W2, b2, z2


# return the accuracy of the neural network model
def test_accuracy(W1, b1, W2, b2, graph, labels):
    nums = int(labels.shape[1])
    z1 = torch.matmul(W1, graph) + b1
    q1 = F.relu(z1)
    p2 = q1
    z2 = torch.matmul(W2, p2) + b2
    cost = common.cross_entropy_with_softmax(labels, z2) / nums
    label = torch.argmax(labels, dim=0)
    pred = torch.argmax(z2, dim=0)
    return (torch.sum(torch.eq(pred, label), dtype=torch.float32).item() / nums, cost)

def objective(graph, labels,
              W1, b1, z1, q1, u1,
              p2, W2, b2, z2, rho,mu):
    p1 = graph
    loss = common.cross_entropy_with_softmax(labels, z2)
    penalty = 0
    # res = 0
    for j in range(1, 3):
        temp1 = locals()['z' + str(j)] - locals()['W' + str(j)].matmul(locals()['p' + str(j)]) - locals()['b' + str(j)]
        temp4 = locals()['W'+str(j)]
        if j<=1:
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
# dataset = cora()
# dataset = pubmed()
# dataset = citeseer()
dataset = amazon_computers()
# dataset = amazon_photo()
# dataset = coauthor_cs()
# initialization
x_train = dataset.x_train
x_train = torch.transpose(x_train, 0, 1)
y_train = dataset.y_train
y_train = torch.transpose(y_train, 0, 1)
x_test = dataset.x_test
x_test = torch.transpose(x_test, 0, 1)
y_test = dataset.y_test
y_test = torch.transpose(y_test, 0, 1)
num_of_neurons = 1000
ITER = 200
index = 0
W1, b1, z1, q1, p2, W2, b2, z2 = Net(x_train,num_of_neurons)
u1 = torch.zeros(q1.shape)
train_acc = np.zeros(ITER)
test_acc = np.zeros(ITER)
train_cost = np.zeros(ITER)
test_cost = np.zeros(ITER)
rho = 1e-5
mu = 10
#cora rho=1e-5 mu=10
#pubmed rho=1e-5 mu=10
#citeseer rho=1e-5 mu=10
#amazon computers rho=1e-5 mu=10
#amazon photo rho=1e-5 mu=10
#coauthor CS rho=1e-5 mu=10
for i in range(ITER):
    pre = time.time()
    print("iter=", i)
    p2 = common.update_p(p2, q1, W2, b2, z2, u1, rho)
    W1 = common.update_W(x_train, b1, z1, W1, rho, mu)
    W2 = common.update_W(p2, b2, z2, W2, rho, mu)
    b1 = common.update_b(x_train, W1, z1, b1, rho)
    b2 = common.update_b(p2, W2, z2, b2, rho)
    z1 = common.update_z(z1, x_train, W1, b1, q1, rho)
    z2 = common.update_zl(p2, W2, b2, y_train, z2, rho)
    q1 = common.update_q(p2, z1, u1, rho)
    r1 = p2- q1
    u1 = u1 + rho * r1
    print("Time per iteration:", time.time() - pre)
    print("rho=", rho)
    (train_acc[i], train_cost[i]) = test_accuracy(W1, b1, W2, b2, x_train, y_train)
    print("training cost:", train_cost[i])
    print("training acc:", train_acc[i])
    (test_acc[i], test_cost[i]) = test_accuracy(W1, b1, W2, b2, x_test, y_test)
    print("test cost:", test_cost[i])
    print("test acc:", test_acc[i])
    obj=objective(x_train, y_train,W1, b1, z1, q1, u1,
              p2, W2, b2, z2, rho, mu)
    linear_r=torch.sum(r1*r1)
    print("obj:",obj.detach().numpy())
    print("linear_r:", linear_r.detach().numpy())
    if i > 2 and train_cost[i] > train_cost[i - 1]-0.001 \
            and train_cost[i - 1] > train_cost[i - 2]-0.001:
            rho = np.minimum(10 * rho, 1)
torch.save(
    {"W1": W1, "b1": b1, "z1": z1, "q1": q1}, \
    'pdadmm_amazon_computers_' + repr(num_of_neurons) + '_2layers.pt')
