# Parallel pdADMM-G


## How to run the code
In this code, we aim to run $N\times M$ layers on *M+1* machines (agents). Here,  *Machine 0* is to generate and distribute the parameters to other *M* machines before the training starts, and during the training process it is responsible for collecting required parameters from each machine and computing accuracy. *Machines 1* to *M* work in parallel: *Machine i* is responsible for training *Layers* $(i-1)\times N + 1$  to $i \times N$ sequentially, $i = 1, 2, \cdots, M$.



1. Modify *config.ini* on each agent.
  - below is an example of *config.ini* for *Machine 0:*
  ```
 [currentMachine]
machine = 0
# machine 0: scheduler

[common]
dataset_name = flickr
;cora, pubmed, citeseer, amazon_computers, amazon_photo, coauthor_cs, ogbn_arxiv, flickr
total_layers = 4
;total number of layers

num_layers_per_machine = 2

iteration = 200
#number of iterations
rho = 1e-4
mu = 0.01
seed_num = 0
neurons = 2000
# number of neurons

plasma_path = ./tmp/plasma
#modify ‘./tmp/plasma’ to an existing path
platform = cpu
#cpu or gpu
chunks = 10
#how many chunks do you want to split the weights

[machine0]
server = 10.65.187.246

[machine1]
server = 10.65.187.225

[machine2]
server = 10.65.187.132

  ```
2. On each agent, run the following command. For the detailed use of Plasma, please visit https://arrow.apache.org/docs/python/plasma.html
```
plasma_store -m 30000000 -s ./tmp/plasma
```
3. On each agent, run the following command:
```
python3 server_pdADMM_G.py
```

4. On each agent, run:
```
python3 client_pdADMM_G.py
```
or

```
python3 client_pdADMM_G_Q.py
```
