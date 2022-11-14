from configparser import ConfigParser
import numpy as np
import time
from tornado.tcpclient import TCPClient
from tornado.ioloop import IOLoop
from tornado import gen,concurrent
from functools import partial
import common
from input_data import cora, pubmed, citeseer, amazon_computers, amazon_photo, coauthor_cs, ogbn_arxiv, flickr, reddit2, reddit, ogbn_products
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
except:
    config.read('config.ini')

machine = config.getint('currentMachine', 'machine')
num_layers_per_machine = config.getint('common', 'num_layers_per_machine')
dataset_name = config['common']['dataset_name']
seed_num = config.getint('common', 'seed_num')
num_of_neurons = config.getint('common', 'neurons')
rho = config.getfloat('common', 'rho')
mu = config.getfloat('common', 'mu')
total_layers = config.getint('common', 'total_layers')
ITER = config.getint('common', 'iteration')
platform = config['common']['platform']
chunks = config.getint('common','chunks')
client = plasma.connect(config['common']['plasma_path'])


logger = logging.getLogger()
logger.setLevel(level=logging.INFO)
handler = logging.FileHandler(str(machine) + '.log')
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
        data = torch.load(os.path.join(BASE_DIR, "pdadmm_g_"+dataset_name+"_" + str(num_of_neurons)+"_"+repr(0) + "_5layers.pt"))
    except:
        data = torch.load("pdadmm_g_"+dataset_name+"_" + str(num_of_neurons)+"_"+repr(0) + "_5layers.pt")

    model = {}
    seed_num = 0
    np.random.seed(seed=seed_num)
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

    # for i in range(1, layers):
    #     model['u' + str(i)] = np.zeros(model['q' + str(i)].shape)

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
    new_list = [i.detach().numpy().astype(np.float32) for i in list]
    global tran_time
    tran_time += (time.time() - tran_start_time)
    return new_list

def update_p(layer, para_dict, time_dict, rho, logger, send_p=False, previous_machine_ip=None,postfix_for_send=None):
    logger.info("Start to compute %s", ('p' + str(layer)))
    p_start_time = time.time()
    para_dict['p' + str(layer)] = common.update_p(para_dict['p' + str(layer)],
                                                  para_dict['q' + str(layer - 1)],
                                                  para_dict['W' + str(layer)],
                                                  para_dict['b' + str(layer)],
                                                  para_dict['z' + str(layer)],
                                                  para_dict['u' + str(layer - 1)], rho)
    time_dict['p'] = time.time() - p_start_time
    logger.info("Finish computing %s", ('p' + str(layer)))
    if send_p:
        logger.info('Start to transfer p into memory')
        p_temp = tensor_to_numpy([para_dict['p' + str(layer)]])[0]
        logger.info("Sending p to previous machine")
        p.apply_async(start_send_splitted_parameter,
                      args=(previous_machine_ip, p_temp, 'p' + str(layer).zfill(2)+postfix_for_send,))
        logger.info("Finish sending p")
    return para_dict['p' + str(layer)], time_dict


def update_W(layer, para_dict, time_dict, rho, mu,scheduler_ip,postfix_for_send, logger):
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

    logger.info('Sending W  to scheduler for accuracy!')
    W_temp = tensor_to_numpy([para_dict['W' + str(layer)]])[0]
    p.apply_async(start_send_whole_parameter, args=(scheduler_ip, W_temp, 'W' + postfix_for_send,))
    logger.info('Finish sending W !')

    return para_dict['W' + str(layer)],time_dict


def update_b(layer, para_dict, time_dict, rho, scheduler_ip,postfix_for_send, logger):
    logger.info("Start to compute %s", ('b' + str(layer)))
    b_start_time = time.time()
    # update b
    para_dict['b' + str(layer)] = common.update_b(para_dict['p' + str(layer)],
                                                  para_dict['W' + str(layer)],
                                                  para_dict['z' + str(layer)],
                                                  para_dict['b' + str(layer)], rho)
    time_dict['b'] = time.time() - b_start_time
    logger.info("Finish computing %s", ('b' + str(layer)))

    logger.info('Sending b  to scheduler for accuracy!')
    b_temp = tensor_to_numpy([para_dict['b' + str(layer)]])[0]
    p.apply_async(start_send_whole_parameter, args=(scheduler_ip, b_temp, 'b' + postfix_for_send,))
    logger.info('Finish sending b !')

    return para_dict['b' + str(layer)], time_dict


def update_zl(layer, para_dict, ytrain, time_dict, rho, logger):
    logger.info("Start to compute %s", ('z' + str(layer)))
    z_start_time = time.time()
    para_dict['z' + str(layer)] = common.update_zl(para_dict['p' + str(layer)],
                                                   para_dict['W' + str(layer)],
                                                   para_dict['b' + str(layer)], ytrain,
                                                   para_dict['z' + str(layer)], rho)
    time_dict['z'] = time.time() - z_start_time
    logger.info("Finish computing %s", ('z' + str(layer)))
    return para_dict['z' + str(layer)], time_dict


def update_z(layer, para_dict, time_dict, rho, logger):
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
    return para_dict['z' + str(layer)], time_dict


def update_q(layer, para_dict, time_dict, rho, logger):
    logger.info("Start to compute %s", ('q' + str(layer)))
    q_start_time = time.time()
    para_dict['q' + str(layer)] = common.update_q(para_dict['p' + str(layer + 1)],
                                                  para_dict['z' + str(layer)],
                                                  para_dict['u' + str(layer)], rho)
    time_dict['q'] = time.time() - q_start_time
    logger.info("Finish computing %s", ('q' + str(layer)))
    return para_dict['q' + str(layer)], time_dict


def update_u(layer, para_dict, time_dict, rho, logger):
    logger.info("Start to compute %s", ('u' + str(layer)))
    u_start_time = time.time()
    para_dict['u' + str(layer)] = para_dict['u' + str(layer)] + rho * (para_dict['p' + str(layer + 1)] - para_dict['q' + str(layer)])
    time_dict['u'] = time.time() - u_start_time
    logger.info("Finish computing %s", ('u' + str(layer)))
    return para_dict['u' + str(layer)], time_dict

# def update_u(u, rho, p, q):
#     return u + rho * (p - q)


if __name__ == '__main__':
    logger.info('dataset: {}'.format(dataset_name))
    logger.info('Number of hidden neuron: {}'.format(num_of_neurons))
    logger.info('Machine number is : %d', machine)
    logger.info('total layer: {}'.format(total_layers))
    scheduler_ip = config['machine0']['server']
    total_machines = int(total_layers / num_layers_per_machine)
    if machine != 0: # not scheduler
        layer_list = [num_layers_per_machine * (machine-1) + i for i in range(1, num_layers_per_machine+1)]
        logger.info('layer list: {}'.format(layer_list))
        if machine != total_machines:
            next_machine_ip = config['machine' + str(machine + 1)]['server']
        if machine > 1:
            previous_machine_ip = config['machine' + str(machine - 1)]['server']

    time_csv = dataset_name + str(machine) + '.csv'

    f = open(time_csv, 'w')
    tran_time = 0

    # init
    avg_time = 0
    avg_compute_time = 0
    avg_communication_time = 0
    if machine == 0:

        p = Pool(10)  # process pool; max num of pool = 10
    else:
        p = Pool(10)
        para_dict = {}

    # schedular machine: send initial parameters to all machines
    if machine == 0:
        # initialization
        logger.info('Start Initialize the Net!')
        rho_count = 0
        if dataset_name == 'cora':
            dataset = cora()
            rho = 1e-4
            mu = 0.01
        elif dataset_name == 'pubmed':
            dataset = pubmed()
            rho = 1e-4
            mu = 1
        elif dataset_name == 'citeseer':
            dataset = citeseer()
            rho = 1e-4
            mu = 0.01
        elif dataset_name == 'amazon_computers':
            dataset = amazon_computers()
            rho = 1e-4
            mu = 0.01
        elif dataset_name == 'amazon_photo':
            dataset = amazon_photo()
            rho = 1e-4
            mu = 0.01
        elif dataset_name == 'coauthor_cs':
            dataset = coauthor_cs()
            rho = 1e-4
            mu = 0.01
        elif dataset_name == 'ogbn_arxiv':
            dataset = ogbn_arxiv()
            rho = 1e-4
            mu = 0.01
        elif dataset_name == 'flickr':
            dataset = flickr()
            rho = 1e-4
            mu = 0.01
        elif dataset_name == 'reddit2':
            dataset = reddit2()
            rho = 1e-4
            mu = 0.01
        elif dataset_name == 'ogbn_products':
            dataset = ogbn_products()
            rho = 1e-4
            mu = 0.01
        else:
            raise ValueError('Please type the correct dataset name.')
        xtrain = dataset.x_train
        xtrain = torch.transpose(xtrain, 0, 1).numpy()
        ytrain = dataset.y_train
        ytrain = torch.transpose(ytrain, 0, 1).numpy()
        xtest = dataset.x_test
        xtest = torch.transpose(xtest, 0, 1).numpy()
        ytest = dataset.y_test
        ytest = torch.transpose(ytest, 0, 1).numpy()
        # model = generate_pretrain_Net(label=ytrain, num_of_neurons=num_of_neurons, layers=total_layers)
        model = generate_Net(xtrain, ytrain, num_of_neurons, total_layers)

        # send parameters to other machines
        machine_parameters = ['p', 'W', 'b', 'z', 'q', 'u']

        # send q of the last layer in the previous machine
        for i in range(1, total_machines):
            ip_address = config['machine' + str(i+1)]['server']
            pre_layer_name = str(i * num_layers_per_machine).zfill(2) + '_000'
            p.apply_async(start_send_whole_parameter, args=(ip_address, model['q' + str(i*num_layers_per_machine)], 'q' + pre_layer_name,))

        # send p of current machine
        machine_1_ip = config['machine1']['server']
        p.apply_async(start_send_whole_parameter, args=(machine_1_ip, xtrain, 'xtrain',))
        if num_layers_per_machine != 1:
            for j in range(2, num_layers_per_machine +1):
                curr_layer_name = str(j).zfill(2) + '_000'
                p.apply_async(start_send_whole_parameter,
                              args=(machine_1_ip, model['p' + str(j)], 'p' + curr_layer_name,))


        for i in range(2, total_machines + 1):
            ip_address = config['machine' + str(i)]['server']
            for j in range((i-1)*num_layers_per_machine+1, i*num_layers_per_machine+1):
                curr_layer_name = str(j).zfill(2) + '_000'
                p.apply_async(start_send_whole_parameter,
                          args=(ip_address, model['p' + str(j)], 'p' + curr_layer_name,))

        # send W, b, z

        params = ['W', 'b', 'z']
        for param in params:

            for i in range(1, total_machines + 1):
                ip_address = config['machine' + str(i)]['server']
                for j in range((i-1)*num_layers_per_machine+1, i*num_layers_per_machine+1):
                    curr_layer_name = str(j).zfill(2) + '_000'
                    p.apply_async(start_send_whole_parameter,
                                  args=(ip_address, model[param + str(j)], param + curr_layer_name,))

        # send q
        for i in range(1, total_machines):
            ip_address = config['machine' + str(i)]['server']
            for j in range((i - 1) * num_layers_per_machine + 1, i * num_layers_per_machine + 1):
                curr_layer_name = str(j).zfill(2) + '_000'
                p.apply_async(start_send_whole_parameter,
                                      args=(ip_address, model['q' + str(j)], 'q' + curr_layer_name,))

        if num_layers_per_machine != 1: # should send q to last machine
            ip_address = config['machine' + str(total_machines)]['server']
            for j in range((total_machines - 1) * num_layers_per_machine + 1, total_layers): # don't send q to the last layer
                curr_layer_name = str(j).zfill(2) + '_000'
                p.apply_async(start_send_whole_parameter,
                                      args=(ip_address, model['q' + str(j)], 'q' + curr_layer_name,))

        # send p of the first layer in the next machine
        for i in range(1, total_machines):
            ip_address = config['machine' + str(i)]['server']
            next_layer_name = str(i * num_layers_per_machine + 1).zfill(2) + '_000'

            p.apply_async(start_send_whole_parameter,
                  args=(ip_address, model['p' + str(i * num_layers_per_machine + 1)], 'p' + next_layer_name,))


        # send y_train to the last layer:
        ip_address = config['machine' + str(total_machines)]['server']
        curr_layer_name = str(total_layers).zfill(2) + '_000'
        p.apply_async(start_send_whole_parameter, args=(ip_address, ytrain, 'ytrain'))

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
        # postfix_for_send = str(machine).zfill(2) + '_' + str(i).zfill(3)
        postfix_for_send =  '_' + str(i).zfill(3)

        # scheduler calculate accuracy
        if machine == 0:
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
                # send rho to all previous machines
                rho_name = 'rho' + str(rho_count)
                for j in range(1, total_machines):
                    logger.info("Sending rho to machine %s", j)
                    p.apply_async(start_send_whole_parameter,
                                  args=(config['machine' + str(i)]['server'], rho, rho_name,))

                    logger.info("Finish sending rho to machine %s!", j)
                rho_count += 1
       # machine 1
        elif machine == 1:
            xtrain_name = 'xtrain'.zfill(10)
            if i == 1: # first iteration
                rho_count = 0
                for j in range(1, num_layers_per_machine+1):
                    curr_postfix = str(j).zfill(2) + '_000|00'
                    next_postfix = str(j+1).zfill(2) + '_000|00'
                    while(1):
                        if check_one_existance(
                                'b' + curr_postfix) and check_one_existance('z' + curr_postfix) and check_one_existance(
                            'W' + curr_postfix) and check_one_existance('q' + curr_postfix) and check_one_existance(
                            'p' + next_postfix):
                            break
                        else:
                            logger.info('Waiting W{}, b{}, z{}, q{}, p{}!'.format(j,j,j,j,j+1))
                            time.sleep(1)
                    if j == 1:
                        while(1):
                            if check_one_existance(xtrain_name):
                                break
                            else:
                                logger.info('Wainting xtrain!')
                                time.sleep(1)
                        para_dict['p1'] = numpy_to_tensor([get_value(xtrain_name)])[0]


                    para_dict['W' + str(j)] = get_value('W' + curr_postfix)
                    para_dict['b' + str(j)] = get_value('b' + curr_postfix)
                    para_dict['z' + str(j)] = get_value('z' + curr_postfix)
                    para_dict['q' + str(j)] = get_value('q' + curr_postfix)
                    para_dict['p' + str(j+1)] = get_value('p' + next_postfix)

                    para_dict['p' + str(j+1)], para_dict['q' + str(j)], para_dict['W' + str(j)], \
                    para_dict['b' + str(j)], \
                    para_dict['z' + str(j)] = numpy_to_tensor(
                        [para_dict['p' + str(j+1)],
                        para_dict['q' + str(j)],
                        para_dict['W' + str(j)],
                        para_dict['b' + str(j)],
                        para_dict['z' + str(j)]])

                    para_dict['u' + str(j)] = torch.zeros_like(para_dict['q' + str(j)])

                    # update parameters
                    if j != 1:
                    # should update p, but don't need to send p, because current layer and previous layer are on the same machine
                        para_dict['p'+str(j)], time_dict = update_p(j, para_dict, time_dict, rho, logger)

                    para_dict['W'+str(j)], time_dict = update_W(j, para_dict, time_dict, rho, mu, scheduler_ip, str(j).zfill(2)+postfix_for_send, logger)
                    para_dict['b'+str(j)], time_dict = update_b(j, para_dict, time_dict, rho, scheduler_ip, str(j).zfill(2)+postfix_for_send, logger)
                    para_dict['z'+str(j)], time_dict = update_z(j, para_dict, time_dict, rho, logger)
                    para_dict['q'+str(j)], time_dict = update_q(j, para_dict, time_dict, rho, logger)
                    para_dict['u'+str(j)], time_dict = update_u(j, para_dict, time_dict, rho, logger)

                    if j == num_layers_per_machine and i != ITER: # should send q and u to next machine
                        logger.info('Sending q and u to next machine!')
                        q_temp, u_temp = tensor_to_numpy(
                            [para_dict['q' + str(j)], para_dict['u' + str(j)]])
                        p.apply_async(start_send_splitted_parameter,
                                      args=(next_machine_ip, q_temp, 'q' + str(j).zfill(2)+postfix_for_send,))
                        p.apply_async(start_send_splitted_parameter,
                                      args=(next_machine_ip, u_temp, 'u' + str(j).zfill(2)+postfix_for_send,))
                        logger.info('Finish sending q and u!')
            else: # iteration > 1
                rho_name = ('rho' + str(rho_count)).zfill(10)
                if check_one_existance(rho_name):
                    rho = np.array(get_value(rho_name))
                    rho = numpy_to_tensor([rho])[0]
                    rho_count += 1

                logger.info('iter={}, rho={}'.format(i, rho))

                for j in range(1, num_layers_per_machine+1):
                    curr_postfix = str(j).zfill(2) + '_' + str(i-1).zfill(3)
                    next_postfix = str(j+1).zfill(2) + '_' + str(i-1).zfill(3)

                    if j != 1:
                        # should update p, but don't need to send p, because current layer and previous layer are on the same machine
                        para_dict['p' + str(j)], time_dict = update_p(j, para_dict, time_dict, rho, logger)


                    para_dict['W' + str(j)], time_dict = update_W(j, para_dict, time_dict, rho, mu, scheduler_ip,
                                                                  str(j).zfill(2) + postfix_for_send, logger)
                    para_dict['b' + str(j)], time_dict = update_b(j, para_dict, time_dict, rho, scheduler_ip,
                                                                  str(j).zfill(2) + postfix_for_send, logger)
                    para_dict['z' + str(j)], time_dict = update_z(j, para_dict, time_dict, rho, logger)


                    if j == num_layers_per_machine: # should receive p from next layer
                        while(1):
                            if check_existance('p'+ next_postfix):
                                break
                            else:
                                logger.info('Waiting p{}!'.format(next_postfix))
                                time.sleep(1)
                                wait_time += 1

                        para_dict['p' + str(j+1)] = aggregate_para('p'+ next_postfix)

                    para_dict['q' + str(j)], time_dict = update_q(j, para_dict, time_dict, rho, logger)
                    para_dict['u' + str(j)], time_dict = update_u(j, para_dict, time_dict, rho, logger)

                    if j == num_layers_per_machine and i != ITER: # should send q and u to next machine
                        logger.info('Sending q and u to next machine!')
                        q_temp, u_temp = tensor_to_numpy(
                            [para_dict['q' + str(j)], para_dict['u' + str(j)]])
                        p.apply_async(start_send_splitted_parameter,
                                      args=(next_machine_ip, q_temp, 'q' + str(j).zfill(2)+postfix_for_send,))
                        p.apply_async(start_send_splitted_parameter,
                                      args=(next_machine_ip, u_temp, 'u' + str(j).zfill(2)+postfix_for_send,))
                        logger.info('Finish sending q and u!')




        else: # machine ID >1
            if i == 1: # first iteration
                rho_count = 0
                for j in range((machine-1) * num_layers_per_machine + 1, machine * num_layers_per_machine + 1):
                    # check existance of first layer in the current machine
                    curr_layer = str(j)
                    pre_layer = str(j-1)
                    next_layer = str(j+1)
                    curr_postfix = curr_layer.zfill(2) + '_000|00'
                    pre_postfix =pre_layer.zfill(2) + '_000|00'
                    next_postfix = next_layer.zfill(2) + '_000|00'

                    if j == (machine-1) * num_layers_per_machine + 1: # the first layer in the current machine
                        while(1):
                            if check_one_existance('p' + curr_postfix) and  check_one_existance('q' + pre_postfix) :
                                break
                            else:
                                logger.info('Waiting p{}, q{} !'.format(curr_postfix, pre_postfix))
                                time.sleep(1)
                        para_dict['p' + curr_layer] = get_value('p' + curr_postfix)
                        para_dict['q' + pre_layer] = get_value('q' + pre_postfix)
                        para_dict['p' + curr_layer], para_dict['q' + pre_layer] = numpy_to_tensor([para_dict['p' + curr_layer], para_dict['q' + pre_layer]])

                        para_dict['u' + pre_layer] = torch.zeros_like(para_dict['q' + pre_layer])



                    while (1):
                        if check_one_existance('W' + curr_postfix) and check_one_existance('b' + curr_postfix) \
                                and check_one_existance('z' + curr_postfix):
                            # and check_one_existance('z' + curr_postfix) and check_one_existance('u' + pre_postfix):
                            break
                        else:
                            # logger.info('Waiting %s and %s!', ('q' + str(layer - 1)), ('u' + str(layer - 1)))
                            logger.info('Waiting   w{}, b{}, z{} !'.format(curr_postfix, curr_postfix, curr_postfix))
                            time.sleep(1)

                    start_time = time.time()
                    para_dict['W' + curr_layer] = get_value('W' + curr_postfix)
                    para_dict['b' + curr_layer] = get_value('b' + curr_postfix)
                    para_dict['z' + curr_layer] = get_value('z' + curr_postfix)

                    para_dict['W' + curr_layer], para_dict[
                        'b' + curr_layer], \
                    para_dict['z' + curr_layer] = numpy_to_tensor([para_dict['W' + curr_layer],
                                                                   para_dict['b' + curr_layer],
                                                                   para_dict['z' + curr_layer]])




                    if j % num_layers_per_machine == 1: # the first layer in the current machine
                        # update p and send p to the previous layer
                        para_dict['p' + curr_layer], time_dict = update_p(int(curr_layer), para_dict, time_dict, rho, logger, send_p=True, previous_machine_ip=previous_machine_ip,postfix_for_send=postfix_for_send)
                    else:
                        para_dict['p' + curr_layer], time_dict = update_p(int(curr_layer), para_dict, time_dict, rho, logger)

                    para_dict['W' + curr_layer], time_dict = update_W(int(curr_layer), para_dict, time_dict, rho, mu, scheduler_ip,
                                                                  curr_layer.zfill(2) + postfix_for_send, logger)
                    para_dict['b' + curr_layer], time_dict = update_b(int(curr_layer), para_dict, time_dict, rho, scheduler_ip,
                                                                  curr_layer.zfill(2) + postfix_for_send, logger)

                    if j != total_layers:
                        # not the last layer; need to receive q and p; update_z instead of update_zl.
                        while(1):
                            if check_one_existance('q'+curr_postfix) and check_one_existance('p'+next_postfix):
                                break
                            else:
                                logger.info('Waiting q{}, p{}!'.format(curr_postfix, next_postfix))
                                time.sleep(1)

                        para_dict['q' + curr_layer] = get_value('q' + curr_postfix)
                        para_dict['p' + next_layer] = get_value('p' + next_postfix)
                        para_dict['q' + curr_layer], para_dict['p' + next_layer] = numpy_to_tensor([para_dict['q' + curr_layer], para_dict['p' + next_layer]])
                        para_dict['u' + curr_layer] = torch.zeros_like(para_dict['q' + curr_layer])

                        # update z, q, u
                        para_dict['z' + curr_layer], time_dict = update_z(j, para_dict, time_dict, rho,
                                                                          logger)
                        para_dict['q' + curr_layer], time_dict = update_q(j, para_dict, time_dict, rho, logger)
                        para_dict['u' + curr_layer], time_dict = update_u(j, para_dict, time_dict, rho, logger)

                        if j % num_layers_per_machine == 0 and i != ITER: # the last layer in the current machine; should send q and u to the next machine
                            logger.info('Sending q and u to next machine!')
                            q_temp, u_temp = tensor_to_numpy(
                                [para_dict['q' + curr_layer], para_dict['u' + curr_layer]])
                            p.apply_async(start_send_splitted_parameter,
                                          args=(next_machine_ip, q_temp, 'q' + curr_layer.zfill(2)+postfix_for_send,))
                            p.apply_async(start_send_splitted_parameter,
                                          args=(next_machine_ip, u_temp, 'u' + curr_layer.zfill(2)+postfix_for_send,))
                            logger.info('Finish sending q and u!')

                    else:  # the last layer; don't need q and u; need ytrain; update_zl instead of update_z
                        ytrain_name = 'ytrain'.zfill(10)
                        while (1):
                            if check_one_existance(ytrain_name):
                                break
                            else:
                                logger.info('Waiting ytrain!')
                                time.sleep(1)
                                wait_time += 1
                        ytrain = get_value(ytrain_name)
                        ytrain = numpy_to_tensor([ytrain])[0]
                        para_dict['z' + curr_layer], time_dict = update_zl(j, para_dict, ytrain, time_dict, rho, logger)


            else: # iteration > 1
                rho_name = ('rho' + str(rho_count)).zfill(10)
                if check_one_existance(rho_name):
                    # rho = np.array(get_value_wo_delete(rho_name))
                    rho = np.array(get_value(rho_name))
                    rho = numpy_to_tensor([rho])[0]
                    rho_count += 1
                # logger.info('rho is : %f', rho)
                logger.info('iter={}, rho={}'.format(i, rho))

                for j in range((machine-1) * num_layers_per_machine + 1, machine * num_layers_per_machine + 1):
                    # check existance of first layer in the current machine
                    curr_layer = str(j)
                    pre_layer = str(j-1)
                    next_layer = str(j+1)
                    curr_postfix = curr_layer.zfill(2) + '_' + str(i-1).zfill(3)
                    pre_postfix =pre_layer.zfill(2) + '_' + str(i-1).zfill(3)
                    next_postfix = next_layer.zfill(2) + '_' + str(i-1).zfill(3)

                    if j % num_layers_per_machine == 1: # the first layer in the current machine; should receive q and u from the previous layer
                        while(1):
                            if check_existance('q'+pre_postfix) and check_existance('u'+pre_postfix):
                                break
                            else:
                                logger.info('Waiting q{} and u{} !'.format(pre_postfix, pre_postfix))
                                time.sleep(1)
                                wait_time += 1

                        para_dict['q'+pre_layer] = aggregate_para('q'+pre_postfix)
                        para_dict['u'+pre_layer] = aggregate_para('u'+pre_postfix)

                        # update p and send p to the previous layer
                        para_dict['p' + curr_layer], time_dict = update_p(int(curr_layer), para_dict, time_dict, rho, logger, send_p=True, previous_machine_ip=previous_machine_ip,postfix_for_send=postfix_for_send)
                    else:
                        para_dict['p' + curr_layer], time_dict = update_p(int(curr_layer), para_dict, time_dict, rho, logger)

                    para_dict['W' + curr_layer], time_dict = update_W(int(curr_layer), para_dict, time_dict, rho, mu, scheduler_ip,
                                                                  curr_layer.zfill(2) + postfix_for_send, logger)
                    para_dict['b' + curr_layer], time_dict = update_b(int(curr_layer), para_dict, time_dict, rho, scheduler_ip,
                                                                  curr_layer.zfill(2) + postfix_for_send, logger)


                    if j % num_layers_per_machine == 0 and j != total_layers:
                        # the last layer of the current machine but not the last layer; need to receive p from the next layer; update_z instead of update_zl.
                        while(1):
                            if check_existance('p'+next_postfix):
                                break
                            else:
                                logger.info('Waiting p{}!'.format(next_postfix))
                                time.sleep(1)

                        para_dict['p' + next_layer] = aggregate_para('p' + next_postfix)

                        # update z, q, u
                        para_dict['z' + curr_layer], time_dict = update_z(j, para_dict, time_dict, rho,
                                                                          logger)
                        para_dict['q' + curr_layer], time_dict = update_q(j, para_dict, time_dict, rho, logger)
                        para_dict['u' + curr_layer], time_dict = update_u(j, para_dict, time_dict, rho, logger)

                        if j % num_layers_per_machine == 0 and i != ITER: # the last layer in the current machine; should send q and u to the next machine
                            logger.info('Sending q and u to next machine!')
                            q_temp, u_temp = tensor_to_numpy(
                                [para_dict['q' + curr_layer], para_dict['u' + curr_layer]])
                            p.apply_async(start_send_splitted_parameter,
                                          args=(next_machine_ip, q_temp, 'q' + curr_layer.zfill(2)+postfix_for_send,))
                            p.apply_async(start_send_splitted_parameter,
                                          args=(next_machine_ip, u_temp, 'u' + curr_layer.zfill(2)+postfix_for_send,))
                            logger.info('Finish sending q and u!')

                    if j == total_layers:  # the last layer; update_zl instead of update_z
                        para_dict['z' + curr_layer], time_dict = update_zl(j, para_dict, ytrain, time_dict, rho, logger)

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
        avg_time += t / ITER
        avg_communication_time += time_dict['wait'] / ITER
        avg_compute_time += time_dict['compute'] / ITER

        if i == ITER:
            p.close()
            p.join()
    logger.info('Average time per iteration: {:.6f}'.format(avg_time))
    logger.info('Average computation time per iteration: {:.6f}'.format(avg_compute_time))
    logger.info('Average communication time per iteration: {:.6f}'.format(avg_communication_time))












