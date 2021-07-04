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
    data = torch.load("pdadmm_q_amazon_photo_"+str(num_of_neurons)+"_2layers.pt")
    W1 = data['W1']
    b1 = data['b1']
    z1 = data['z1']
    q1 = data['q1']
    p2 = q1
    W2 = torch.eye(num_of_neurons)
    b2 = torch.zeros(size=(num_of_neurons, 1))
    z2 = torch.matmul(W2, p2) + b2
    q2 = F.relu(z2)
    p3 = q2
    W3 = torch.eye(num_of_neurons)
    b3 = torch.zeros(size=(num_of_neurons, 1))
    z3 = torch.matmul(W3, p3) + b3
    q3 = F.relu(z3)
    p4 = q3
    W4 = torch.eye(num_of_neurons)
    b4 = torch.zeros(size=(num_of_neurons, 1))
    z4 = torch.matmul(W4, p4) + b4
    q4 = F.relu(z4)
    p5 = q4
    W5 = torch.normal(size=(dataset.num_classes, num_of_neurons), mean=0, std=0.1, requires_grad=True)
    b5 = torch.normal(size=(dataset.num_classes, 1), mean=0, std=0.1, requires_grad=True)
    z5 = torch.zeros(size=(dataset.num_classes, graph.size()[1]), requires_grad=True)

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
# dataset = cora()
# dataset = pubmed()
# dataset = citeseer()
# dataset = amazon_computers()
dataset = amazon_photo()
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
W1, b1, z1, q1, p2, W2, b2, z2, q2, p3, W3, b3, z3, q3, p4, W4, b4, z4, q4, p5, W5, b5, z5 = Net(x_train,
                                                                                                 num_of_neurons)
#
#
# data = torch.load('pdadmm_q_amazon_photo_1000_0-99epochs_5layers.pt')
# W1 = data['W1']
# b1 = data['b1']
# z1 = data['z1']
# q1 = data['q1']
# u1 = data['u1']
# p2 = data['p2']
# W2 = data['W2']
# b2 = data['b2']
# z2 = data['z2']
# q2 = data['q2']
# u2 = data['u2']
# p3 = data['p3']
# W3 = data['W3']
# b3 = data['b3']
# z3 = data['z3']
# q3 = data['q3']
# u3 = data['u3']
# p4 = data['p4']
# W4 = data['W4']
# b4 = data['b4']
# z4 = data['z4']
# q4 = data['q4']
# u4 = data['u4']
# p5 = data['p5']
# W5 = data['W5']
# b5 = data['b5']
# z5 = data['z5']
# rho = data['rho']

u1 = torch.zeros(q1.shape)
u2 = torch.zeros(q2.shape)
u3 = torch.zeros(q3.shape)
u4 = torch.zeros(q4.shape)
train_acc = np.zeros(ITER)
test_acc = np.zeros(ITER)
train_cost = np.zeros(ITER)
test_cost = np.zeros(ITER)
delta = np.linspace(-1, 20, 22)
delta = torch.from_numpy(delta).unsqueeze(dim=-1).float()
#cora rho=1e-5 mu=0.1
#pubmed photo rho=1e-5 mu=0.1
#citeseer rho=1e-5 mu=0.1
#amazon computers rho=1e-5 mu=0.1
#amazon photo rho=1e-5 mu=0.1
#coauthor cs rho=1e-5 mu=0.1
rho = 1e-5
mu = 0.1
for i in range(ITER):
    pre = time.time()
    print("iter=", i)
    p2 = common.update_p_quantize(p2, q1, W2, b2, z2, u1, rho,delta)
    p3 = common.update_p_quantize(p3, q2, W3, b3, z3, u2, rho,delta)
    p4 = common.update_p_quantize(p4, q3, W4, b4, z4, u3, rho,delta)
    p5 = common.update_p_quantize(p5, q4, W5, b5, z5, u4, rho,delta)
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
    q1 = common.update_q(p2, z1, u1, rho)
    q2 = common.update_q(p3, z2, u2, rho)
    q3 = common.update_q(p4, z3, u3, rho)
    q4 = common.update_q(p5, z4, u4, rho)
    r1 = p2-q1
    r2 = p3-q2
    r3 = p4-q3
    r4 = p5-q4
    u1 = u1 + rho * r1
    u2 = u2 + rho * r2
    u3 = u3 + rho * r3
    u4 = u4 + rho * r4
    print("Time per iteration:", time.time() - pre)
    print("rho=", rho)
    (train_acc[i], train_cost[i]) = test_accuracy(W1, b1, W2, b2, W3, b3, W4, b4, W5, b5, x_train, y_train)
    print("training cost:", train_cost[i])
    print("training acc:", train_acc[i])
    (test_acc[i], test_cost[i]) = test_accuracy(W1, b1, W2, b2, W3, b3, W4, b4, W5, b5, x_test, y_test)
    print("test cost:", test_cost[i])
    print("test acc:", test_acc[i])
    obj=objective(x_train, y_train, W1, b1, z1, q1, u1,
              p2, W2, b2, z2, q2,u2,
              p3, W3, b3, z3, q3,u3,
              p4, W4, b4, z4, q4,u4,
              p5, W5, b5, z5,rho,mu)
    linear_r=torch.sum(r1*r1+r2*r2+r3*r3+r4*r4)
    print("obj:",obj.detach().numpy())
    print("linear_r:", linear_r.detach().numpy())
    if i > 2 and train_cost[i] > train_cost[i - 1]-0.001 \
        and train_cost[i - 1] > train_cost[i - 2]-0.001 :
           rho = np.minimum(10 * rho, 1)
torch.save(
    {"W1": W1, "b1": b1, "z1": z1, "q1": q1, "u1": u1,
     "p2": p2, "W2": W2, "b2": b2, "z2": z2, "q2": q2, "u2": u2,
     "p3": p3, "W3": W3, "b3": b3, "z3": z3, "q3": q3, "u3": u3,
     "p4": p4, "W4": W4, "b4": b4, "z4": z4, "q4": q4, "u4": u4,
     "p5": p5, "W5": W5, "b5": b5, "z5": z5,
     "rho": rho}, \
    'pdadmm_q_amazon_photo_' + repr(num_of_neurons) + '_5layers.pt')
