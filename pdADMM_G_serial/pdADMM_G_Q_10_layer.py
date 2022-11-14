import torch
import numpy as np
import sys
import time

if not sys.warnoptions:
    import warnings

    warnings.simplefilter("ignore")

import common
from input_data import cora, pubmed, citeseer, amazon_computers, amazon_photo, coauthor_cs, coauthor_physics, ogbn_arxiv, flickr
import torch.nn.functional as F
import os



def Net(label, num_of_neurons, seed):
    data = torch.load("pdadmm_g_q_"+dataset_name+"_" + str(num_of_neurons)+"_"+repr(seed) + "_5layers.pt")
    W1 = data['W1'].to(device)
    b1 = data['b1'].to(device)
    z1 = data['z1'].to(device)
    q1 = data['q1'].to(device)
    p2 = data['p2'].to(device)
    W2 = data['W2'].to(device)
    b2 = data['b2'].to(device)
    z2 = data['z2'].to(device)
    q2 = data['q2'].to(device)
    p3 = data['p3'].to(device)
    W3 = data['W3'].to(device)
    b3 = data['b3'].to(device)
    z3 = data['z3'].to(device)
    q3 = data['q3'].to(device)
    p4 = data['p4'].to(device)
    W4 = data['W4'].to(device)
    b4 = data['b4'].to(device)
    z4 = data['z4'].to(device)
    q4 = data['q4'].to(device)
    del data

    p5 = q4
    W5 = torch.eye(num_of_neurons, requires_grad=True).to(device)
    b5 = torch.zeros(size=(num_of_neurons, 1), requires_grad=True).to(device)
    z5 = torch.matmul(W5, p5) + b5
    q5 = F.relu(z5)
    p6 = q5
    W6 = torch.eye(num_of_neurons, requires_grad=True).to(device)
    b6 = torch.zeros(size=(num_of_neurons, 1), requires_grad=True).to(device)
    z6 = torch.matmul(W6, p6) + b6
    q6 = F.relu(z6)
    p7 = q6
    W7 = torch.eye(num_of_neurons, requires_grad=True).to(device)
    b7 = torch.zeros(size=(num_of_neurons, 1), requires_grad=True).to(device)
    z7 = torch.matmul(W7, p7) + b7
    q7 = F.relu(z7)
    p8 = q7
    W8 = torch.eye(num_of_neurons, requires_grad=True).to(device)
    b8 = torch.zeros(size=(num_of_neurons, 1), requires_grad=True).to(device)
    z8 = torch.matmul(W8, p8) + b8
    q8 = F.relu(z8)
    p9 = q8
    W9 = torch.eye(num_of_neurons, requires_grad=True).to(device)
    b9 = torch.zeros(size=(num_of_neurons, 1), requires_grad=True).to(device)
    z9 = torch.matmul(W9, p9) + b9
    q9 = F.relu(z9)
    p10 = q9
    W10 = torch.eye(n=num_classes, m=num_of_neurons, requires_grad=True).to(device)
    b10 = torch.zeros(size=(num_classes, 1), requires_grad=True).to(device)
    imask = torch.eq(label, torch.zeros(size=label.size()).to(device))
    z10 = torch.where(imask, -1 * torch.ones_like(label).to(device), 1 * torch.ones_like(label).to(device))
    return W1, b1, z1, q1, p2, W2, b2, z2, q2, p3, W3, b3, z3, q3, p4, W4, b4, z4, q4, p5, W5, b5, z5, q5, p6, W6, b6, z6, q6, p7, W7, b7, z7, q7, p8, W8, b8, z8, q8, p9, W9, b9, z9, q9, p10, W10, b10, z10


# return the accuracy of the neural network model
def test_accuracy(W1, b1, W2, b2, W3, b3, W4, b4, W5, b5, W6, b6, W7, b7, W8, b8, W9, b9, W10, b10, graph, labels):
    nums = int(labels.shape[1])
    p2 = F.relu(torch.matmul(W1, graph) + b1)
    p3 = F.relu(torch.matmul(W2, p2) + b2)
    p4 = F.relu(torch.matmul(W3, p3) + b3)
    p5 = F.relu(torch.matmul(W4, p4) + b4)
    p6 = F.relu(torch.matmul(W5, p5) + b5)
    p7 = F.relu(torch.matmul(W6, p6) + b6)
    p8 = F.relu(torch.matmul(W7, p7) + b7)
    p9 = F.relu(torch.matmul(W8, p8) + b8)
    p10 = F.relu(torch.matmul(W9, p9) + b9)
    z10 = torch.matmul(W10, p10) + b10
    cost = common.cross_entropy_with_softmax(labels, z10) / nums
    label = torch.argmax(labels, dim=0)
    pred = torch.argmax(z10, dim=0)
    return (torch.sum(torch.eq(pred, label), dtype=torch.float32).item() / nums, cost)


def objective(graph, labels,
              W1, b1, z1, q1, u1,
              p2, W2, b2, z2, q2, u2,
              p3, W3, b3, z3, q3, u3,
              p4, W4, b4, z4, q4, u4,
              p5, W5, b5, z5, q5, u5,
              p6, W6, b6, z6, q6, u6,
              p7, W7, b7, z7, q7, u7,
              p8, W8, b8, z8, q8, u8,
              p9, W9, b9, z9, q9, u9,
              p10, W10, b10, z10, rho,mu):
    p1 = graph
    loss = common.cross_entropy_with_softmax(labels, z10)
    penalty = 0
    # res = 0
    for j in range(1, 11):
        temp1 = locals()['z' + str(j)] - locals()['W' + str(j)].matmul(locals()['p' + str(j)]) - locals()['b' + str(j)]
        temp4 = locals()['W'+str(j)]
        if j<=9:
            temp2 = locals()['q' + str(j)] - F.relu(locals()['z' + str(j)])
            temp3 = locals()['p' + str(j + 1)] - locals()['q' + str(j)]
            penalty += rho / 2 * torch.sum(temp1 * temp1) + rho / 2 * torch.sum(temp2 * temp2) \
                       + torch.sum((rho / 2 * temp3 + locals()['u' + str(j)]) * temp3) + mu / 2 * torch.sum(
                temp4 * temp4)
        else:
            temp2=0
            temp3=0
            penalty += rho / 2 * torch.sum(temp1 * temp1)  + mu / 2 * torch.sum(temp4 * temp4)
    obj = loss + penalty
    return obj

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")
dataset_name = 'cora'
# cora, pubmed, citeseer, amazon_computers, amazon_photo, coauthor_cs,coauthor_physics,flickr, ogbn_arxiv

print('dataset: {}'.format(dataset_name))
if dataset_name == 'cora':
    dataset = cora()
    rho = 1e-4
    mu = 0.01
    # mu = 0.1
elif dataset_name == 'pubmed':
    dataset = pubmed()
    rho=1e-3
    mu=0.01
elif dataset_name == 'citeseer':
    dataset = citeseer()
    rho=1e-3
    mu=0.01
elif dataset_name == 'amazon_computers':
    dataset = amazon_computers()
    rho=1e-3
    mu=0.001
elif dataset_name == 'amazon_photo':
    dataset = amazon_photo()
    rho=1e-3
    mu=0.001
elif dataset_name == 'coauthor_cs':
    dataset = coauthor_cs()
    rho=1e-2
    mu=0.01
elif dataset_name == 'coauthor_physics':
    dataset = coauthor_physics()
    rho=1e-2
    mu=0.01
elif dataset_name == 'ogbn_arxiv':
    dataset = ogbn_arxiv()
    rho = 1e-3
    mu = 0.01
elif dataset_name == 'flickr':
    dataset = flickr()
    rho = 1e-4
    mu = 0
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
seeds = [0, 100, 200, 300, 400]
delta = np.linspace(-1, 2, 1000)  # small datasets
# delta1 = np.linspace(0,1.98,100) # flickr
# delta2 = np.linspace(2,10,101)
# delta1 = np.linspace(0,5,101) # ogbn_arxiv
# delta2 = np.linspace(5.3,11,20)
# #
# delta1 = torch.from_numpy(delta1).unsqueeze(dim=-1).float()
# delta2 = torch.from_numpy(delta2).unsqueeze(dim=-1).float()
# delta = torch.cat([delta1, delta2], dim=0)

rho_back=rho
for seed in seeds:
    rho = rho_back
    print('-'*20)
    print('seed: {}'.format(seed))
    min_epoch = 0
    train_acc = np.zeros(ITER)
    val_acc = np.zeros(ITER)
    train_cost = np.zeros(ITER)
    val_cost = np.zeros(ITER)
    linear_r = np.zeros(ITER)
    obj = np.zeros(ITER)
    if os.path.exists('pdadmm_g_q_'+dataset_name+'_' + repr(num_of_neurons) + '_'+repr(seed) +'_10layers.pt')\
            and os.path.exists('pdadmm_g_q_'+dataset_name+'_' + repr(num_of_neurons)+'_'+repr(seed) + '_10layers_acc.pt'):
        f = torch.load('pdadmm_g_q_'+dataset_name+'_' + repr(num_of_neurons) + '_'+repr(seed) +'_10layers.pt')
        print('loaded checkpoint')
        min_epoch = f['i'] + 1
        # min_epoch = 0
        rho = f['rho']
        W1 = f['W1'].to(device)
        b1 = f['b1'].to(device)
        z1 = f['z1'].to(device)
        q1 = f['q1'].to(device)
        for j in range(2,10):
            locals()['p' + str(j)]  = f['p'+str(j)].to(device)
            locals()['W' + str(j)]  = f['W'+str(j)].to(device)
            locals()['b' + str(j)]  = f['b'+str(j)].to(device)
            locals()['z' + str(j)]  = f['z'+str(j)].to(device)
            locals()['q' + str(j)]  = f['q'+str(j)].to(device)
        p10 = f['p10'] .to(device)
        W10 = f['W10'] .to(device)
        b10 = f['b10'] .to(device)
        z10 = f['z10'].to(device)
        del  f

        p = torch.load('pdadmm_g_q_'+dataset_name+'_' + repr(num_of_neurons)+'_'+repr(seed) + '_10layers_acc.pt')



        if p["train_acc"].shape[0] < ITER:
            train_acc[0:p["train_acc"].shape[0]] = p["train_acc"]
            train_cost[0:p["train_acc"].shape[0]] = p["train_cost"]
            val_acc[0:p["train_acc"].shape[0]] = p["val_acc"]
            val_cost[0:p["train_acc"].shape[0]] = p["val_cost"]
            linear_r[0:p["train_acc"].shape[0]] = p["linear_r"]
            obj[0:p["train_acc"].shape[0]] = p["obj"]
            t_train = p['t_train']
        elif p["train_acc"].shape[0] > ITER:
            train_acc = p["train_acc"][0:p["train_acc"].shape[0]]
            train_cost = p["train_cost"][0:p["train_acc"].shape[0]]
            val_acc = p["val_acc"][0:p["train_acc"].shape[0]]
            val_cost = p["val_cost"][0:p["train_acc"].shape[0]]
            linear_r = p["linear_r"][0:p["train_acc"].shape[0]]
            obj = p["obj"][0:p["train_acc"].shape[0]]
            t_train = p['t_train']
        else:
            linear_r = p["linear_r"]
            obj = p["obj"]
            train_acc = p["train_acc"]
            train_cost = p["train_cost"]
            val_acc = p["val_acc"]
            val_cost = p["val_cost"]
            t_train = p['t_train']

        del p
    else:
        W1, b1, z1, q1, p2, W2, b2, z2, q2, p3, W3, b3, z3, q3, p4, W4, b4, z4, q4, p5, W5, b5, z5, q5, p6, W6, b6, z6, q6, \
    p7, W7, b7, z7, q7, p8, W8, b8, z8, q8, p9, W9, b9, z9, q9, p10, W10, b10, z10 = Net(y_train, num_of_neurons, seed)

        # train_acc = np.zeros(ITER)
        # val_acc = np.zeros(ITER)
        # train_cost = np.zeros(ITER)
        # val_cost = np.zeros(ITER)
        # linear_r = np.zeros(ITER)
        # obj = np.zeros(ITER)

    u1 = torch.zeros(q1.shape).to(device)
    u2 = torch.zeros(q2.shape).to(device)
    u3 = torch.zeros(q3.shape).to(device)
    u4 = torch.zeros(q4.shape).to(device)
    u5 = torch.zeros(q5.shape).to(device)
    u6 = torch.zeros(q6.shape).to(device)
    u7 = torch.zeros(q7.shape).to(device)
    u8 = torch.zeros(q8.shape).to(device)
    u9 = torch.zeros(q9.shape).to(device)


    t_train = 0
    for i in range(min_epoch, ITER):
        pre = time.time()
        print('-'*20)
        print("iter=", i)
        # p2 = common.update_p_quantize(p2, q1, W2, b2, z2, u1, rho,delta)
        # p3 = common.update_p_quantize(p3, q2, W3, b3, z3, u2, rho,delta)
        # p4 = common.update_p_quantize(p4, q3, W4, b4, z4, u3, rho,delta)
        # p5 = common.update_p_quantize(p5, q4, W5, b5, z5, u4, rho,delta)
        # p6 = common.update_p_quantize(p6, q5, W6, b6, z6, u5, rho,delta)
        # p7 = common.update_p_quantize(p7, q6, W7, b7, z7, u6, rho,delta)
        # p8 = common.update_p_quantize(p8, q7, W8, b8, z8, u7, rho,delta)
        # p9 = common.update_p_quantize(p9, q8, W9, b9, z9, u8, rho,delta)
        # p10 = common.update_p_quantize(p10, q9, W10, b10, z10, u9, rho,delta)

        p2 = common.update_p(p2, q1, W2, b2, z2, u1, rho)
        p3 = common.update_p(p3, q2, W3, b3, z3, u2, rho)
        p4 = common.update_p(p4, q3, W4, b4, z4, u3, rho)
        p5 = common.update_p(p5, q4, W5, b5, z5, u4, rho)
        p6 = common.update_p(p6, q5, W6, b6, z6, u5, rho)
        p7 = common.update_p(p7, q6, W7, b7, z7, u6, rho)
        p8 = common.update_p(p8, q7, W8, b8, z8, u7, rho)
        p9 = common.update_p(p9, q8, W9, b9, z9, u8, rho)
        p10 = common.update_p(p10, q9, W10, b10, z10, u9, rho)

        W1 = common.update_W(x_train, b1, z1, W1, rho, mu)
        W2 = common.update_W(p2, b2, z2, W2, rho, mu)
        W3 = common.update_W(p3, b3, z3, W3, rho, mu)
        W4 = common.update_W(p4, b4, z4, W4, rho, mu)
        W5 = common.update_W(p5, b5, z5, W5, rho, mu)
        W6 = common.update_W(p6, b6, z6, W6, rho, mu)
        W7 = common.update_W(p7, b7, z7, W7, rho, mu)
        W8 = common.update_W(p8, b8, z8, W8, rho, mu)
        W9 = common.update_W(p9, b9, z9, W9, rho, mu)
        W10 = common.update_W(p10, b10, z10, W10, rho, mu)
        b1 = common.update_b(x_train, W1, z1, b1, rho)
        b2 = common.update_b(p2, W2, z2, b2, rho)
        b3 = common.update_b(p3, W3, z3, b3, rho)
        b4 = common.update_b(p4, W4, z4, b4, rho)
        b5 = common.update_b(p5, W5, z5, b5, rho)
        b6 = common.update_b(p6, W6, z6, b6, rho)
        b7 = common.update_b(p7, W7, z7, b7, rho)
        b8 = common.update_b(p8, W8, z8, b8, rho)
        b9 = common.update_b(p9, W9, z9, b9, rho)
        b10 = common.update_b(p10, W10, z10, b10, rho)
        z1 = common.update_z(z1, x_train, W1, b1, q1, rho)
        z2 = common.update_z(z2, p2, W2, b2, q2, rho)
        z3 = common.update_z(z3, p3, W3, b3, q3, rho)
        z4 = common.update_z(z4, p4, W4, b4, q4, rho)
        z5 = common.update_z(z5, p5, W5, b5, q5, rho)
        z6 = common.update_z(z6, p6, W6, b6, q6, rho)
        z7 = common.update_z(z7, p7, W7, b7, q7, rho)
        z8 = common.update_z(z8, p8, W8, b8, q8, rho)
        z9 = common.update_z(z9, p9, W9, b9, q9, rho)
        z10 = common.update_zl(p10, W10, b10, y_train, z10, rho)

        # q1 = common.update_q(p2, z1, u1, rho)
        # q2 = common.update_q(p3, z2, u2, rho)
        # q3 = common.update_q(p4, z3, u3, rho)
        # q4 = common.update_q(p5, z4, u4, rho)
        # q5 = common.update_q(p6, z5, u5, rho)
        # q6 = common.update_q(p7, z6, u6, rho)
        # q7 = common.update_q(p8, z7, u7, rho)
        # q8 = common.update_q(p9, z8, u8, rho)
        # q9 = common.update_q(p10, z9, u9, rho)

        q1 = common.update_q_quantize(p2, z1, u1, rho, delta)
        q2 = common.update_q_quantize(p3, z2, u2, rho, delta)
        q3 = common.update_q_quantize(p4, z3, u3, rho, delta)
        q4 = common.update_q_quantize(p5, z4, u4, rho, delta)
        q5 = common.update_q_quantize(p6, z5, u5, rho, delta)
        q6 = common.update_q_quantize(p7, z6, u6, rho, delta)
        q7 = common.update_q_quantize(p8, z7, u7, rho, delta)
        q8 = common.update_q_quantize(p9, z8, u8, rho, delta)
        q9 = common.update_q_quantize(p10, z9, u9, rho, delta)
        r1 = p2 - q1
        r2 = p3 - q2
        r3 = p4 - q3
        r4 = p5 - q4
        r5 = p6 - q5
        r6 = p7 - q6
        r7 = p8 - q7
        r8 = p9 - q8
        r9 = p10- q9
        u1 = u1 + rho * r1
        u2 = u2 + rho * r2
        u3 = u3 + rho * r3
        u4 = u4 + rho * r4
        u5 = u5 + rho * r5
        u6 = u6 + rho * r6
        u7 = u7 + rho * r7
        u8 = u8 + rho * r8
        u9 = u9 + rho * r9

        t_train += time.time() - pre
        print("Time per iteration:", time.time() - pre)
        print("rho=", rho)
        (train_acc[i], train_cost[i]) = test_accuracy(W1, b1, W2, b2, W3, b3, W4, b4, W5, b5, W6, b6, W7, b7, W8, b8, W9,
                                                      b9, W10, b10, x_train, y_train)
        print("training cost:", train_cost[i])
        print("training acc:", train_acc[i])
        (val_acc[i], val_cost[i]) = test_accuracy(W1, b1, W2, b2, W3, b3, W4, b4, W5, b5, W6, b6, W7, b7, W8, b8, W9, b9,
                                                  W10, b10, x_val, y_val)
        print("validation cost:", val_cost[i])
        print("validation acc:", val_acc[i])
        # obj[i] = objective(x_train, y_train,
        #           W1, b1, z1, q1, u1,
        #           p2, W2, b2, z2, q2, u2,
        #           p3, W3, b3, z3, q3, u3,
        #           p4, W4, b4, z4, q4, u4,
        #           p5, W5, b5, z5, q5, u5,
        #           p6, W6, b6, z6, q6, u6,
        #           p7, W7, b7, z7, q7, u7,
        #           p8, W8, b8, z8, q8, u8,
        #           p9, W9, b9, z9, q9, u9,
        #           p10, W10, b10, z10, rho,mu)
        # print('objective:', obj[i])
        linear_r[i] = torch.sum((r1*r1).cpu()+(r2*r2).cpu()+(r3*r3).cpu()+(r4*r4).cpu()+(r5*r5).cpu()+(r6*r6).cpu()+(r7*r7).cpu()+(r8*r8).cpu()+(r9*r9).cpu())
        print("res:", linear_r[i])
        if i > 20 and i < 50 and train_cost[i] > train_cost[i - 1] - 0.001 \
                  and train_cost[i - 1] > train_cost[i - 2]:
             rho = np.minimum(10 * rho, 0.01)

        if i > 50 and train_cost[i] > train_cost[i - 1] - 0.001 \
                  and train_cost[i - 1] > train_cost[i - 2]:
             rho = np.minimum(10 * rho, 1e-3)


        torch.save(
            {"i":i,'rho':rho,
                "W1": W1, "b1": b1, "z1": z1, "q1": q1, "u1": u1,
             "p2": p2, "W2": W2, "b2": b2, "z2": z2, "q2": q2, "u2": u2,
             "p3": p3, "W3": W3, "b3": b3, "z3": z3, "q3": q3, "u3": u3,
             "p4": p4, "W4": W4, "b4": b4, "z4": z4, "q4": q4, "u4": u4,
             "p5": p5, "W5": W5, "b5": b5, "z5": z5, "q5": q5, "u5": u5,
             "p6": p6, "W6": W6, "b6": b6, "z6": z6, "q6": q6, "u6": u6,
             "p7": p7, "W7": W7, "b7": b7, "z7": z7, "q7": q7, "u7": u7,
             "p8": p8, "W8": W8, "b8": b8, "z8": z8, "q8": q8, "u8": u8,
             "p9": p9, "W9": W9, "b9": b9, "z9": z9, "q9": q9, "u9": u9,
             "p10": p10, "W10": W10, "b10": b10, "z10": z10},
            './pdadmm_g_q_'+dataset_name+'_' + repr(num_of_neurons) + '_'+repr(seed) +'_10layers.pt')

        torch.save(
            {"linear_r": linear_r, "obj": obj,
             "train_acc": train_acc, "train_cost": train_cost, "val_acc": val_acc, "val_cost": val_cost,
              't_train': t_train},
            './pdadmm_g_q_' + dataset_name + '_' + repr(num_of_neurons) + '_' + repr(seed) + '_10layers_acc.pt')

    t_train = t_train / ITER
    print('Average training time per iteration: {}'.format(t_train))
    print("training cost:", train_cost[ITER - 1])
    print("training acc:", train_acc[ITER - 1])
    print("validation cost:", val_cost[ITER - 1])
    print("validation acc:", val_acc[ITER - 1])
    (test_acc, test_cost) = test_accuracy(W1, b1, W2, b2, W3, b3, W4, b4, W5, b5, W6, b6, W7, b7, W8, b8, W9, b9,
                                              W10, b10, x_test, y_test)
    test_cost=test_cost.item()
    print("test cost:", test_cost)
    print("test acc:", test_acc)

    torch.save(
        { "linear_r": linear_r, "obj": obj,
          "train_acc": train_acc, "train_cost": train_cost, "val_acc": val_acc, "val_cost": val_cost,
          "test_acc": test_acc, "test_cost": test_cost, 't_train':t_train},
        './pdadmm_g_q_'+dataset_name+'_' + repr(num_of_neurons)+'_'+repr(seed) + '_10layers_acc.pt')
print('finish training!')