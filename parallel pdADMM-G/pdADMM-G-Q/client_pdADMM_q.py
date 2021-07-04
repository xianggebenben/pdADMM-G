from configparser import ConfigParser
import numpy as np
import time
from tornado.tcpclient import TCPClient
from tornado.ioloop import IOLoop
from tornado import gen,concurrent
from functools import partial
import common
from input_data import cora, pubmed, citeseer, amazon_photo, amazon_computers, coauthor_cs
import pickle
import codecs
import pyarrow.plasma as plasma
# from brain_plasma import Brain
import logging
import torch

from multiprocessing import Pool
import gc
import threading
import multiprocessing
import csv
import os
import sys

if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")

# read config file
config = ConfigParser()
try:
    # config.read(os.path.dirname(os.getcwd())+'/config.ini')
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    config.read(os.path.join(BASE_DIR, 'config.ini'))
    layer = config.getint('currentLayer', 'layer')
except:
    config.read('config.ini')
    layer = config.getint('currentLayer', 'layer')

seed_num = config.getint('common', 'seed_num')
num_of_neurons = config.getint('common', 'neurons')
rho = config.getfloat('common', 'rho')
mu = config.getfloat('common', 'mu')
total_layers = config.getint('common', 'total_layers')
ITER = config.getint('common', 'iteration')
platform = config['common']['platform']
chunks = config.getint('common','chunks')
# brain = Brain(path=config['common']['plasma_path'])
# brain.set_namespace('default')
client = plasma.connect(config['common']['plasma_path'])


logger = logging.getLogger()
logger.setLevel(level=logging.INFO)
handler = logging.FileHandler(str(layer) + '.log')
formatter = logging.Formatter('%(asctime)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)


sh = logging.StreamHandler()
sh.setFormatter(formatter)
logger.addHandler(sh)

sentinel = b'---end---'


def obj_to_pickle_string(x):
    return codecs.encode(pickle.dumps(x), "base64").decode()

def relu(x):
    return np.maximum(x, 0)
def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)
# return cross entropy
def cross_entropy(label, prob):
    loss = -np.sum(label * np.log(prob))
    return loss
# return the cross entropy loss function
def cross_entropy_with_softmax(label, z):
    prob = softmax(z)
    loss = cross_entropy(label, prob)
    return loss


def generate_Net(graph, label, num_of_neurons, layers):
    model = {}
    seed_num = 0
    np.random.seed(seed=seed_num)
    model['W1'] = np.random.normal(0, 0.1, size=(num_of_neurons, graph.shape[0])).astype(np.float32)
    np.random.seed(seed=seed_num)
    model['b1'] = np.random.normal(0, 0.1, size=(num_of_neurons, 1)).astype(np.float32)
    model['z1'] = np.matmul(model['W1'], graph) + model['b1']
    model['q1'] = relu(model['z1'])

    # middle layers
    for i in range(2, layers):
        model['p' + str(i)] = model['q' + str(i - 1)]
        np.random.seed(seed=seed_num)
        model['W' + str(i)] = np.random.normal(0, 0.1, size=(num_of_neurons, num_of_neurons)).astype(np.float32)
        np.random.seed(seed=seed_num)
        model['b' + str(i)] = np.random.normal(0, 0.1, size=(num_of_neurons, 1)).astype(np.float32)
        model['z' + str(i)] = np.matmul(model['W' + str(i)], model['q' + str(i - 1)]) + model['b' + str(i)]
        model['q' + str(i)] = relu(model['z' + str(i)])

    # last layer
    model['p' + str(layers)] = model['q' + str(layers - 1)]
    np.random.seed(seed=seed_num)
    model['W' + str(layers)] = np.random.normal(0, 0.1, size=(dataset.num_classes, num_of_neurons))
    np.random.seed(seed=seed_num)
    model['b' + str(layers)] = np.random.normal(0, 0.1, size=(dataset.num_classes, 1))
    model['z' + str(layers)] = np.ones(label.shape)
    model['z' + str(layers)][label == 0] = -1
    model['z' + str(layers)][label == 1] = 1

    for i in range(1, layers):
        model['u' + str(i)] = np.zeros(model['q' + str(i)].shape)

    return model


def generate_pretrain_Net(label, num_of_neurons, layers):
    try:
        data = torch.load(os.path.join(BASE_DIR, 'pdadmm_cora_2000_5layers.pt'))
    except:
        data = torch.load('pdadmm_cora_2000_5layers.pt')

    model = {}
    # seed_num = 0
    # np.random.seed(seed=seed_num)
    model['W1'] = data['W1'].detach().numpy().astype(np.float32)
    model['b1'] = data['b1'].detach().numpy().astype(np.float32)
    model['z1'] = data['z1'].detach().numpy().astype(np.float32)
    model['q1'] = data['q1'].detach().numpy().astype(np.float32)
    for i in range(2, 5):
        model['p' + str(i)] = data['p' + str(i)].detach().numpy().astype(np.float32)
        model['W' + str(i)] = data['W' + str(i)].detach().numpy().astype(np.float32)
        model['b' + str(i)] = data['b' + str(i)].detach().numpy().astype(np.float32)
        model['z' + str(i)] = data['z' + str(i)].detach().numpy().astype(np.float32)
        model['q' + str(i)] = data['q' + str(i)].detach().numpy().astype(np.float32)

    for i in range(5, layers):
        model['p' + str(i)] = model['q' + str(i - 1)]
        np.random.seed(seed=seed_num)
        model['W' + str(i)] = np.eye(num_of_neurons, dtype=np.float32)
        np.random.seed(seed=seed_num)
        model['b' + str(i)] = np.zeros(shape=(num_of_neurons, 1)).astype(np.float32)
        model['z' + str(i)] = np.matmul(model['W' + str(i)], model['q' + str(i - 1)]) + model['b' + str(i)]
        model['q' + str(i)] = relu(model['z' + str(i)])

    # last layer
    model['p' + str(layers)] = model['q' + str(layers - 1)]
    model['W' + str(layers)] = torch.eye(n=dataset.num_classes, m=num_of_neurons).detach().numpy().astype(
        np.float32)
    model['b' + str(layers)] = np.zeros(shape=(dataset.num_classes, 1)).astype(np.float32)
    model['z' + str(layers)] = np.ones(label.shape)
    model['z' + str(layers)][label == 0] = -1
    model['z' + str(layers)][label == 1] = 1

    for i in range(1, layers):
        model['u' + str(i)] = np.zeros(model['q' + str(i)].shape)

    return model


def test_accuracy(para_dict, graph, labels):
    temp_dict = {}
    nums = labels.shape[1]

    temp_dict['z1'] = np.matmul(para_dict['W01'], graph) + para_dict['b01']
    temp_dict['q1'] = relu(temp_dict['z1'])
    for i in range(2, total_layers):

        temp_dict['z'+str(i)] = np.matmul(para_dict['W'+str(i).zfill(2)],temp_dict['q'+str(i-1)]) + para_dict['b'+str(i).zfill(2)]
        temp_dict['q'+str(i)] = relu(temp_dict['z'+str(i)])

    temp_dict['z'+str(total_layers)] = \
        np.matmul(para_dict['W'+str(total_layers).zfill(2)], temp_dict['q'+str(total_layers-1)]) \
        + para_dict['b'+str(total_layers).zfill(2)]
    cost = cross_entropy_with_softmax(labels, temp_dict['z'+str(total_layers)]) / nums
    label = np.argmax(labels, axis=0)
    pred = np.argmax(temp_dict['z'+str(total_layers)], axis=0)
    return np.sum(np.equal(pred, label)) / nums, cost


async def send_parameter(ip, data):
    stream = await TCPClient().connect(ip, 8888)  # host: ip; port: 8888
    await stream.write(data)  # Asynchronously write the given data to this stream
    # await gen.sleep(0)


async def send_splitted_parameter(ip, data, parameter):
    logger.info('Sent parameter to %s ', ip)
    all_works = []
    chunked_array = np.vsplit(data, chunks)

    for index, arr in enumerate(chunked_array):
        new_para = parameter + '|' + str(index + 1).zfill(2)

        logger.info('Sent parameter %s to %s ', new_para, ip)
        all_works.append(send_parameter(ip, pickle.dumps(arr) + str.encode(new_para) + sentinel))
    await gen.multi(all_works)


# For the initialization, we use multiprocessing to send the whole data.
async def send_whole_parameter(ip, data, parameter):
    stream = await TCPClient().connect(ip, 8888)
    if parameter != 'xtrain' and parameter != 'ytrain' and parameter != 'xtest' and parameter != 'ytest' \
            and parameter != "rho0" and parameter != "rho1" and parameter != "rho2" and parameter != "rho3":
        new_para = parameter + '|00'
    else:
        new_para = parameter.zfill(10)
    logger.info('Sent parameter %s to %s', new_para, ip)
    await stream.write(pickle.dumps(data) + str.encode(new_para) + sentinel)
    # gen.sleep(10)


def start_send_whole_parameter(ip, data, parameter):
    # if layer != 0:
    #     data = tensor_to_numpy([data])[0]
    IOLoop.current().run_sync(lambda: send_whole_parameter(ip, data, parameter))
    # time.sleep(1)



def start_send_splitted_parameter(ip, data, parameter):
    # if layer != 0:
    #     data = tensor_to_numpy([data])[0]
    logger.info("start to send splitted para : %s", parameter)
    IOLoop.current().run_sync(lambda: send_splitted_parameter(ip, data, parameter))
    # time.sleep(1)


def check_existance(para_name):
    for i in range(chunks):
        if not client.contains(plasma_id(para_name + '|' + str(i + 1).zfill(2))):
            return False
    return True


def check_one_existance(para_name):
    return client.contains(plasma_id(para_name))


def aggregate_para(para_name):
    all = []
    for i in range(chunks):
        all.append(get_value(para_name + '|' + str(i + 1).zfill(2)))
        # client.delete([plasma_id(para_name+'|'+str(i+1).zfill(2))])
    aggregated_para = numpy_to_tensor([np.concatenate(all)])[0]
    return aggregated_para


def update_u(u, rho, p, q):
    return u + rho * (p - q)


# id in client.list()
def plasma_id(name):
    return plasma.ObjectID(10 * b'0' + str.encode(name))


def get_value(name):
    value = np.array(client.get(plasma_id(name)))
    delete_value(name)
    return value


def get_value_wo_delete(name):
    value = np.array(client.get(plasma_id(name)))
    # delete_value(name)
    return value


def delete_value(name):
    client.delete([plasma_id(name)])


def numpy_to_tensor(list):
    tran_start_time = time.time()
    new_list = [torch.from_numpy(i).float() for i in list]
    global tran_time
    tran_time += (time.time() - tran_start_time)
    return new_list


def tensor_to_numpy(list):
    tran_start_time = time.time()
    new_list = [i.numpy().astype(np.float32) for i in list]
    global tran_time
    tran_time += (time.time() - tran_start_time)
    return new_list


def tensor_to_numpy_int(list):
    tran_start_time = time.time()
    new_list = [i.numpy().astype(np.int8) for i in list]
    global tran_time
    tran_time += (time.time() - tran_start_time)
    return new_list


if __name__ == '__main__':
    # print('Layer number is : ',layer)
    logger.info('Layer number is : %d', layer)
    para_dict = {}
    scheduler_ip = config['layer0']['server']
    if layer != total_layers:
        next_layer_ip = config['layer' + str(layer + 1)]['server']
    if layer > 1:
        previous_layer_ip = config['layer' + str(layer - 1)]['server']

    time_csv = str(layer) + '.csv'
    f = open(time_csv, 'w')
    tran_time = 0
    delta = np.linspace(-1, 20, 22)
    delta = torch.from_numpy(delta).unsqueeze(dim=-1).float()

    # init
    if layer == 0:
        avg_time = 0
        p = Pool(8)  # process pool; max num of pool = 10
    else:
        p = Pool(8)
        avg_time = 0

    # schedular layer: send initial parameters to all layers
    if layer == 0:
        # initialization
        logger.info('Start Initialize the Net!')
        rho_count = 0
        dataset = cora()
        # dataset = pubmed()
        # dataset = citeseer()
        # dataset = amazon_computers()
        # dataset = amazon_photo()
        # dataset = coauthor_cs()
        xtrain = dataset.x_train
        xtrain = torch.transpose(xtrain, 0, 1).numpy()
        ytrain = dataset.y_train
        ytrain = torch.transpose(ytrain, 0, 1).numpy()
        xtest = dataset.x_test
        xtest = torch.transpose(xtest, 0, 1).numpy()
        ytest = dataset.y_test
        ytest = torch.transpose(ytest, 0, 1).numpy()
        model = generate_pretrain_Net(label=ytrain, num_of_neurons=num_of_neurons, layers=total_layers)
        # model = generate_Net(xtrain, ytrain, num_of_neurons, total_layers)

        # send parameters to other layers
        layer_parameters = ['p', 'W', 'b', 'z', 'u', 'q']


        # time.sleep(10)

        # send initialized parameters to middle layers
        for i in range(2, total_layers + 1):
            ip_address = config['layer' + str(i)]['server']

            pre_layer_name = str(i - 1).zfill(2) + '_000'
            next_layer_name = str(i + 1).zfill(2) + '_000'
            curr_layer_name = str(i).zfill(2) + '_000'

            # send q and u of previous layer
            p.apply_async(start_send_whole_parameter, args=(ip_address, model['q' + str(i - 1)], 'q' + pre_layer_name,))
            p.apply_async(start_send_whole_parameter, args=(ip_address, model['u' + str(i - 1)], 'u' + pre_layer_name,))

            # send y_train to the last layer:
            if i == total_layers:
                p.apply_async(start_send_whole_parameter, args=(ip_address, ytrain, 'ytrain'))
                for k in layer_parameters[:-2]:
                    p.apply_async(start_send_whole_parameter,
                                  args=(ip_address, model[k + str(i)], k + curr_layer_name,))
                # time.sleep(5)

            # send rest parameters to layers except the last layer
            else:
                # send p of next layer
                p.apply_async(start_send_whole_parameter,
                              args=(ip_address, model['p' + str(i + 1)], 'p' + next_layer_name,))

                for j in layer_parameters:
                    p.apply_async(start_send_whole_parameter,
                                  args=(ip_address, model[j + str(i)], j + curr_layer_name,))
                # time.sleep(5)
                # send initialized W1,b1,z1,q1,u1,p2 to layer 1
                # initialization
        layer_1_ip = config['layer1']['server']
        # IOLoop.current().run_sync(lambda:send_whole_parameter(config['layer2']['server'], xtrain, 'xtrain'))
        p.apply_async(start_send_whole_parameter, args=(layer_1_ip, xtrain, 'xtrain',))
        p.apply_async(start_send_whole_parameter, args=(layer_1_ip, model['W1'], 'W01_000',))
        p.apply_async(start_send_whole_parameter, args=(layer_1_ip, model['b1'], 'b01_000',))
        p.apply_async(start_send_whole_parameter, args=(layer_1_ip, model['z1'], 'z01_000',))
        p.apply_async(start_send_whole_parameter, args=(layer_1_ip, model['q1'], 'q01_000',))
        p.apply_async(start_send_whole_parameter, args=(layer_1_ip, model['u1'], 'u01_000',))
        p.apply_async(start_send_whole_parameter, args=(layer_1_ip, model['p2'], 'p02_000',))
        time.sleep(10)
        # p.close()  # close the pool so that new tasks cannot get in
        # p.join()  # wait till all tasks finish
        logger.info('Finished Initialization')

    for i in range(1, ITER + 1):
        logger.info('=========================== Iter %d ===========================', i)
        logger.info('rho is : %f', rho)
        time_dict = dict()
        time_dict['iteration'] = i
        start_time = time.time()
        tran_time = 0
        wait_time = 0
        time_send_para_for_training = 0
        postfix_for_send = str(layer).zfill(2) + '_' + str(i).zfill(3)

        # scheduler calculate accuracy
        if layer == 0:
            postfix = str(i).zfill(3) + '|00'
            paras_name_for_accuracy = []
            train_acc = np.zeros(ITER)
            test_acc = np.zeros(ITER)
            train_cost = np.zeros(ITER)
            test_cost = np.zeros(ITER)

            wait = 0
            for l in range(1, total_layers + 1):
                paras_name_for_accuracy.append('W' + str(l).zfill(2) + '_' + postfix)
                paras_name_for_accuracy.append('b' + str(l).zfill(2) + '_' + postfix)

            while (1):
                res = []
                for para in paras_name_for_accuracy:
                    res.append(check_one_existance(para))
                if all(res):
                    # get all needed values
                    para_acc = {}
                    for para in paras_name_for_accuracy:
                        para_acc[para.split('_')[0]] = get_value(para)
                    train_acc[i-1], train_cost[i-1] = test_accuracy(para_acc, xtrain, ytrain)
                    test_acc[i-1], test_cost[i-1] = test_accuracy(para_acc, xtest, ytest)
                    break
                else:
                    time.sleep(1)
                    wait_time += 1
                    if wait_time > 5 * 60:
                        train_acc[i-1], train_cost[i-1] = [0, 0]
                        test_acc[i-1], test_cost[i-1] = [0, 0]
                        break

            logger.info('Iteration %d : train accuracy %f cost %f', i, train_acc[i-1], train_cost[i-1])
            logger.info('Iteration %d : test accuracy %f cost %f', i, test_acc[i-1], test_cost[i-1])
            if i > 3 and train_cost[i - 1] > train_cost[i - 2] - 0.001 \
                    and train_cost[i - 2] > train_cost[i - 3] - 0.001 and rho < 1:
                rho = np.minimum(10 * rho, 1)
                # send rho to all previous layers
                rho_name = 'rho' + str(rho_count)
                for j in range(1, total_layers):
                    logger.info("Sending rho to layer %s", j)
                    p.apply_async(start_send_whole_parameter,
                                  args=(config['layer' + str(i)]['server'], rho, rho_name,))

                    logger.info("Finish sending rho to layer %s!", j)
                rho_count += 1
        # layer 1
        elif layer == 1:
            xtrain_name = 'xtrain'.zfill(10)
            if i == 1:
                curr_postfix = '01_000|00'
                next_postfix = '02_000|00'
                rho_count = 0

                # wait for initialization
                while (1):
                    if check_one_existance(xtrain_name) and check_one_existance(
                            'b' + curr_postfix) and check_one_existance('z' + curr_postfix) and check_one_existance(
                            'W' + curr_postfix) and check_one_existance('q' + curr_postfix) and check_one_existance(
                            'p' + next_postfix) and check_one_existance('u' + curr_postfix):
                        break
                    else:
                        logger.info('Waiting x_train, W1, b1, z1, q1, u1, p2!')
                        time.sleep(1)
                        wait_time += 1

                xtrain = np.array(get_value(xtrain_name))
                W1 = get_value('W' + curr_postfix)
                b1 = get_value('b' + curr_postfix)
                z1 = get_value('z' + curr_postfix)
                # q1 = get_value('q' + curr_postfix)
                # u1 = get_value('u' + curr_postfix)
                # p2 = get_value('p' + next_postfix)
                q1 = get_value_wo_delete('q' + curr_postfix)
                u1 = get_value_wo_delete('u' + curr_postfix)
                p2 = get_value_wo_delete('p' + next_postfix)
                xtrain, b1, W1, z1, q1, u1, p2 = numpy_to_tensor(
                                                    [xtrain, b1, W1, z1, q1, u1, p2])

                # update parameters
                logger.info("Start to compute W1")
                w_start_time = time.time()
                W1 = common.update_W(xtrain, b1, z1, W1, rho, mu)
                time_dict['w'] = time.time() - w_start_time
                logger.info("Finish computing W1")

                logger.info("Start to compute b1")
                b_start_time = time.time()
                b1 = common.update_b(xtrain, W1, z1, b1, rho)
                time_dict['b'] = time.time() - b_start_time
                logger.info("Finish computing b1")

                logger.info("Start to compute z1")
                z_start_time = time.time()
                z1 = common.update_z(z1, xtrain, W1, b1, q1, rho)
                time_dict['z'] = time.time() - z_start_time
                logger.info("Finish computing z1")

                logger.info('Start to compute q1')
                q_start_time = time.time()
                q1 = common.update_q(p2, z1, u1, rho)
                time_dict['q'] = time.time() - q_start_time
                logger.info('Finish computing q1')

                logger.info('Start to compute u1')
                u_start_time = time.time()
                u1 = update_u(u1, rho, p2, q1)
                time_dict['u'] = time.time() - u_start_time
                logger.info('Finish computing u1')

                # send  parameters to neighbor layers
                logger.info('Sending q and u to next layer!')
                q_temp, u_temp = tensor_to_numpy([q1, u1])
                p.apply_async(start_send_splitted_parameter, args=(next_layer_ip, q_temp, 'q01_' + str(i).zfill(3),))
                p.apply_async(start_send_splitted_parameter, args=(next_layer_ip, u_temp, 'u01_' + str(i).zfill(3),))
                logger.info('Finish sending q and u!')

            # iteration > 1: only receive parameters from neighbor layers
            else:
                rho_name = ('rho' + str(rho_count)).zfill(10)
                if check_one_existance(rho_name):
                    rho = np.array(get_value_wo_delete(rho_name))
                    rho = numpy_to_tensor([rho])[0]
                    rho_count += 1

                logger.info('iter={}, rho={}'.format(i, rho))
                logger.info("Start to compute W1")
                w_start_time = time.time()
                W1 = common.update_W(xtrain, b1, z1, W1, rho, mu)
                time_dict['w'] = time.time() - w_start_time
                logger.info("Finish computing W1")

                logger.info("Start to compute b1")
                b_start_time = time.time()
                b1 = common.update_b(xtrain, W1, z1, b1, rho)
                time_dict['b'] = time.time() - b_start_time
                logger.info("Finish computing b1")

                logger.info("Start to compute z1")
                z_start_time = time.time()
                z1 = common.update_z(z1, xtrain, W1, b1, q1, rho)
                time_dict['z'] = time.time() - z_start_time
                logger.info("Finish computing z1")
                # receive parameters from neighbor layers

                while (1):
                    if check_existance('p02_' + str(i).zfill(3)):
                        p2 = aggregate_para('p02_' + str(i).zfill(3))
                        break
                    else:
                        logger.info('Waiting %s !', ('p02_' + str(i).zfill(3)))
                        time.sleep(1)
                        wait_time += 1
                logger.info("Start to compute q1")
                q_start_time = time.time()
                q1 = common.update_q(p2, z1, u1, rho)
                time_dict['q'] = time.time() - q_start_time
                logger.info("Finish computing q1")

                logger.info("Start to compute u1")
                u_start_time = time.time()
                u1 = update_u(u1, rho, p2, q1)
                time_dict['u'] = time.time() - u_start_time
                logger.info("Finish computing u1")

                if i != ITER:
                    logger.info('Sending q and u to next layer!')
                    q_temp, u_temp = tensor_to_numpy([q1, u1])

                    p.apply_async(start_send_splitted_parameter,
                                  args=(next_layer_ip, q_temp, 'q01_' + str(i).zfill(3),))
                    p.apply_async(start_send_splitted_parameter,
                                  args=(next_layer_ip, u_temp, 'u01_' + str(i).zfill(3),))
                    logger.info('Finish sending q and u!')

            # for each iteration: send W1, b1 to scheduler
            second_time = time.time()
            logger.info('Sending W and b to scheduler for accuracy!')

            W_temp, b_temp = tensor_to_numpy([W1, b1])
            p.apply_async(start_send_whole_parameter, args=(scheduler_ip, W_temp, 'W01_' + str(i).zfill(3),))
            p.apply_async(start_send_whole_parameter, args=(scheduler_ip, b_temp, 'b01_' + str(i).zfill(3),))

            time_send_para_for_training += time.time() - second_time
            logger.info('Finish sending W and b!')

        # current_layer > 1
        else:
            ytrain_name = 'ytrain'.zfill(10)
            # iteration 1: wait for initialization
            if i == 1:
                rho_count = 0
                curr_postfix = str(layer).zfill(2) + '_000|00'
                next_postfix = str(layer + 1).zfill(2) + '_000|00'
                pre_postfix = str(layer - 1).zfill(2) + '_000|00'

                while (1):
                    if check_one_existance('p' + curr_postfix) and check_one_existance('q' + pre_postfix)\
                            and check_one_existance('W' + curr_postfix) and check_one_existance('b' + curr_postfix)\
                            and check_one_existance('z' + curr_postfix) and check_one_existance('u' + pre_postfix):
                        break
                    else:
                        logger.info('Waiting %s and %s!', ('q' + str(layer - 1)), ('u' + str(layer - 1)))
                        time.sleep(1)
                        wait_time += 1
                # para_dict['p' + str(layer)] = get_value('p' + curr_postfix)
                # para_dict['q' + str(layer - 1)] = get_value('q' + pre_postfix)
                para_dict['p' + str(layer)] = get_value_wo_delete('p' + curr_postfix)
                para_dict['q' + str(layer - 1)] = get_value_wo_delete('q' + pre_postfix)
                para_dict['W' + str(layer)] = get_value('W' + curr_postfix)
                para_dict['b' + str(layer)] = get_value('b' + curr_postfix)
                para_dict['z' + str(layer)] = get_value('z' + curr_postfix)
                # para_dict['u' + str(layer - 1)] = get_value('u' + pre_postfix)
                para_dict['u' + str(layer - 1)] = get_value_wo_delete('u' + pre_postfix)
                # para_dict['p' + str(layer + 1)] = get_value('p' + next_postfix)


                # numpy to tensor
                para_dict['p' + str(layer)],  para_dict['q' + str(layer - 1)],  para_dict['W' + str(layer)], para_dict['b' + str(layer)], \
                para_dict['z' + str(layer)], para_dict['u' + str(layer - 1)] = numpy_to_tensor([para_dict['p' + str(layer)],
                 para_dict['q' + str(layer - 1)],
                 para_dict['W' + str(layer)],
                 para_dict['b' + str(layer)],
                 para_dict['z' + str(layer)],
                 para_dict['u' + str(layer - 1)]])

                if layer != total_layers:
                    # para_dict['q' + str(layer)] = get_value('q' + curr_postfix)
                    # para_dict['u' + str(layer)] = get_value('u' + curr_postfix)
                    para_dict['q' + str(layer)] = get_value_wo_delete('q' + curr_postfix)
                    para_dict['u' + str(layer)] = get_value_wo_delete('u' + curr_postfix)
                    para_dict['q' + str(layer)], para_dict['u' + str(layer)] = numpy_to_tensor(
                        [para_dict['q' + str(layer)], para_dict['u' + str(layer)]])

                # update p and send p to the previous layer
                logger.info("Start to compute %s", ('p' + str(layer)))
                p_start_time = time.time()

                para_dict['p' + str(layer)] = common.update_p_quantize(para_dict['p' + str(layer)],
                                                              para_dict['q' + str(layer - 1)],
                                                              para_dict['W' + str(layer)],
                                                              para_dict['b' + str(layer)],
                                                              para_dict['z' + str(layer)],
                                                              para_dict['u' + str(layer - 1)], rho, delta)
                time_dict['p'] = time.time() - p_start_time
                logger.info("Finish computing %s", ('p' + str(layer)))
                logger.info('Start to transfer p into memory')
                p_temp = tensor_to_numpy_int([para_dict['p' + str(layer)]])[0]
                logger.info("Sending p to previous layer")
                p.apply_async(start_send_splitted_parameter,
                              args=(previous_layer_ip, p_temp, 'p' + postfix_for_send,))
                logger.info("Finish sending p")

                logger.info("Start to compute %s", ('W' + str(layer)))
                w_start_time = time.time()
                # update W
                para_dict['W' + str(layer)] = common.update_W(para_dict['p' + str(layer)],
                                                              para_dict['b' + str(layer)],
                                                              para_dict['z' + str(layer)],
                                                              para_dict['W' + str(layer)],
                                                              rho, mu)
                time_dict['w'] = time.time() - w_start_time
                logger.info("Finish computing %s", ('W' + str(layer)))

                logger.info("Start to compute %s", ('b' + str(layer)))
                b_start_time = time.time()
                # update b
                para_dict['b' + str(layer)] = common.update_b(para_dict['p' + str(layer)],
                                                              para_dict['W' + str(layer)],
                                                              para_dict['z' + str(layer)],
                                                              para_dict['b' + str(layer)], rho)
                time_dict['b'] = time.time() - b_start_time
                logger.info("Finish computing %s", ('b' + str(layer)))

                # last layer; iteration=1
                if layer == total_layers:
                    while (1):
                        if check_one_existance(ytrain_name):
                            break
                        else:
                            logger.info('Waiting ytrain!')
                            time.sleep(1)
                            wait_time += 1
                    ytrain = get_value(ytrain_name)
                    ytrain = numpy_to_tensor([ytrain])[0]
                    logger.info("Start to compute %s", ('z' + str(layer)))
                    z_start_time = time.time()
                    para_dict['z' + str(layer)] = common.update_zl(para_dict['p' + str(layer)],
                                                                   para_dict['W' + str(layer)],
                                                                   para_dict['b' + str(layer)], ytrain,
                                                                   para_dict['z' + str(layer)], rho)
                    time_dict['z'] = time.time() - z_start_time
                    logger.info("Finish computing %s", ('z' + str(layer)))
                else:
                    rho_name = ('rho' + str(rho_count)).zfill(10)
                    if check_one_existance(rho_name):
                        rho = np.array(get_value_wo_delete(rho_name))
                        rho = numpy_to_tensor([rho])[0]
                        rho_count += 1
                    logger.info('iter={}, rho={}'.format(i, rho))
                    # update z
                    logger.info("Start to compute %s", ('z' + str(layer)))
                    z_start_time = time.time()
                    para_dict['z' + str(layer)] = common.update_z(para_dict['z' + str(layer)],
                                                                  para_dict['p' + str(layer)],
                                                                  para_dict['W' + str(layer)],
                                                                  para_dict['b' + str(layer)],
                                                                  para_dict['q' + str(layer)], rho)
                    time_dict['z'] = time.time() - z_start_time
                    logger.info("Finish computing %s", ('z' + str(layer)))

                    # update q
                    while (1):
                        if check_one_existance('p' + next_postfix):
                            # para_dict['p' + str(layer + 1)] = get_value('p' + next_postfix)
                            para_dict['p' + str(layer + 1)] = get_value_wo_delete('p' + next_postfix)
                            para_dict['p' + str(layer + 1)] = numpy_to_tensor([para_dict['p' + str(layer + 1)]])[0]
                            break
                        else:
                            logger.info("Waiting for %s ", ('p' + str(layer + 1)))
                            time.sleep(1)
                            wait_time += 1

                    logger.info("Start to compute %s", ('q' + str(layer)))
                    q_start_time = time.time()
                    para_dict['q' + str(layer)] = common.update_q(para_dict['p' + str(layer + 1)],
                                                                  para_dict['z' + str(layer)],
                                                                  para_dict['u' + str(layer)], rho)
                    time_dict['q'] = time.time() - q_start_time
                    logger.info("Finish computing %s", ('q' + str(layer)))

                    # update u
                    logger.info("Start to compute %s", ('u' + str(layer)))
                    u_start_time = time.time()
                    para_dict['u' + str(layer)] = update_u(para_dict['u' + str(layer)], rho,
                                                           para_dict['p' + str(layer + 1)], para_dict['q' + str(layer)])
                    time_dict['u'] = time.time() - u_start_time
                    logger.info("Finish computing %s", ('u' + str(layer)))

                    # send q and u
                    logger.info('Sending q and u to next layer!')
                    q_temp, u_temp = tensor_to_numpy([para_dict['q' + str(layer)], para_dict['u' + str(layer)]])
                    p.apply_async(start_send_splitted_parameter, args=(next_layer_ip, q_temp, 'q' + postfix_for_send,))
                    p.apply_async(start_send_splitted_parameter, args=(next_layer_ip, u_temp, 'u' + postfix_for_send,))
                    logger.info('Finish sending q and u!')

            # iteration > 1
            else:
                rho_name = ('rho' + str(rho_count)).zfill(10)
                if check_one_existance(rho_name):
                    rho = np.array(get_value_wo_delete(rho_name))
                    rho = numpy_to_tensor([rho])[0]
                    rho_count += 1
                # logger.info('rho is : %f', rho)
                logger.info('iter={}, rho={}'.format(i, rho))
                # check q and u of previous layer arrived
                pre_layer_q = 'q' + str(layer - 1).zfill(2) + '_' + str(i - 1).zfill(3)
                pre_layer_u = 'u' + str(layer - 1).zfill(2) + '_' + str(i - 1).zfill(3)
                while (1):
                    if check_existance(pre_layer_q) and check_existance(pre_layer_u):
                        break
                    else:
                        logger.info('Waiting for %s and %s!', pre_layer_q, pre_layer_u)
                        time.sleep(1)
                        wait_time += 1
                # update p
                logger.info("Start to compute %s", ('p' + str(layer)))
                p_start_time = time.time()
                para_dict['p' + str(layer)] = common.update_p_quantize(para_dict['p' + str(layer)], aggregate_para(pre_layer_q),
                                                              para_dict['W' + str(layer)], para_dict['b' + str(layer)],
                                                              para_dict['z' + str(layer)], aggregate_para(pre_layer_u),
                                                              rho, delta)
                time_dict['p'] = time.time() - p_start_time
                logger.info("Finish computing %s", ('p' + str(layer)))
                logger.info("Sending p to previous layer")
                p_temp = tensor_to_numpy_int([para_dict['p' + str(layer)]])[0]
                p.apply_async(start_send_splitted_parameter, args=(previous_layer_ip, p_temp, 'p' + postfix_for_send,))
                logger.info("Finish sending p")

                # update W
                logger.info("Start to compute %s", ('W' + str(layer)))
                w_start_time = time.time()
                para_dict['W' + str(layer)] = common.update_W(para_dict['p' + str(layer)], para_dict['b' + str(layer)],
                                                              para_dict['z' + str(layer)], para_dict['W' + str(layer)],
                                                              rho, mu)
                time_dict['w'] = time.time() - w_start_time
                logger.info("Finish computing %s", ('W' + str(layer)))

                # update b
                logger.info("Start to compute %s", ('b' + str(layer)))
                b_start_time = time.time()
                para_dict['b' + str(layer)] = common.update_b(para_dict['p' + str(layer)], para_dict['W' + str(layer)],
                                                              para_dict['z' + str(layer)], para_dict['b' + str(layer)],
                                                              rho)
                time_dict['b'] = time.time() - b_start_time
                logger.info("Finish computing %s", ('b' + str(layer)))

                if layer == total_layers:
                    # update z
                    logger.info("Start to compute %s", ('z' + str(layer)))
                    z_start_time = time.time()
                    para_dict['z' + str(layer)] = common.update_zl(para_dict['p' + str(layer)],
                                                                   para_dict['W' + str(layer)],
                                                                   para_dict['b' + str(layer)], ytrain,
                                                                   para_dict['z' + str(layer)], rho)
                    time_dict['z'] = time.time() - z_start_time
                    logger.info("Finish computing %s", ('z' + str(layer)))
                else:
                    # update z
                    logger.info("Start to compute %s", ('z' + str(layer)))
                    z_start_time = time.time()
                    para_dict['z' + str(layer)] = common.update_z(para_dict['z' + str(layer)],
                                                                  para_dict['p' + str(layer)],
                                                                  para_dict['W' + str(layer)],
                                                                  para_dict['b' + str(layer)],
                                                                  para_dict['q' + str(layer)], rho)
                    time_dict['z'] = time.time() - z_start_time
                    logger.info("Finish computing %s", ('z' + str(layer)))

                    # update q
                    while (1):
                        if check_existance('p' + str(layer + 1).zfill(2) + '_' + str(i).zfill(3)):
                            break
                        else:
                            # print('Waiting for %s !' % ('p'+str(layer+1)+'_'+ str(i)))
                            logger.info('Waiting for %s !', ('p' + str(layer + 1).zfill(2) + '_' + str(i).zfill(3)))
                            time.sleep(1)
                            wait_time += 1
                    logger.info("Start to compute %s", ('q' + str(layer)))
                    aggregated_p = aggregate_para('p' + str(layer + 1).zfill(2) + '_' + str(i).zfill(3))
                    q_start_time = time.time()
                    para_dict['q' + str(layer)] = common.update_q(aggregated_p, para_dict['z' + str(layer)],
                                                                  para_dict['u' + str(layer)], rho)
                    time_dict['q'] = time.time() - q_start_time
                    logger.info("Finish computing %s", ('q' + str(layer)))

                    # update u
                    logger.info("Start to compute %s", ('u' + str(layer)))
                    u_start_time = time.time()
                    para_dict['u' + str(layer)] = update_u(para_dict['u' + str(layer)], rho, aggregated_p,
                                                           para_dict['q' + str(layer)])
                    time_dict['u'] = time.time() - u_start_time
                    logger.info("Finish computing %s", ('u' + str(layer)))
                    if i != ITER:
                        logger.info('Sending q and u to next layer!')
                        q_temp, u_temp = tensor_to_numpy([para_dict['q' + str(layer)], para_dict['u' + str(layer)]])
                        p.apply_async(start_send_splitted_parameter,
                                      args=(next_layer_ip, q_temp, 'q' + postfix_for_send,))
                        p.apply_async(start_send_splitted_parameter,
                                      args=(next_layer_ip, u_temp, 'u' + postfix_for_send,))
                        logger.info('Finish sending q and u!')

            # for all layer>1: send W1, b1 to scheduler
            second_time = time.time()
            logger.info('Sending W and b to scheduler for accuracy!')
            W_temp, b_temp = tensor_to_numpy([para_dict['W' + str(layer)], para_dict['b' + str(layer)]])
            p.apply_async(start_send_whole_parameter, args=(scheduler_ip, W_temp, 'W' + postfix_for_send,))
            p.apply_async(start_send_whole_parameter, args=(scheduler_ip, b_temp, 'b' + postfix_for_send,))
            time_send_para_for_training += time.time() - second_time
            logger.info('Finish sending W and b!')

        time_dict['compute'] = time.time() - start_time - time_send_para_for_training - wait_time
        time_dict['wait'] = wait_time
        time_dict['total_time'] = time.time() - start_time - time_send_para_for_training
        time_dict['tran_time'] = tran_time
        dw = csv.DictWriter(f, time_dict.keys())
        if i == 1:
            dw.writeheader()
        dw.writerow(time_dict)

        t = time.time() - start_time - time_send_para_for_training
        logger.info('Iteration %d takes %f ', i, (t))
        logger.info('Iteration %d wait time %f ', i, wait_time)
        logger.info('Iteration %d compute time %f ', i,
                    (time.time() - start_time - wait_time - time_send_para_for_training))
        avg_time += t/ITER

        if i == ITER:
            p.close()
            p.join()
    logger.info('Average time per iteration: {}'.format(avg_time))