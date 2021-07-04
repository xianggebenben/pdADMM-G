# Parallel pdADMM-G


## How to run the code
In this code, *N+1* agents (denoted as *layer0*, *layer1*...*layerN* in *config.ini* respectively) are needed to run an *N-layer* model. 
 Specifically, we update the parameters of layer *i* on the *i-th* agent.
Before the training starts, the extra agent *layer0* is used to generate and distribute the parameters for each layer. 
During the training process it is responsible for collecting required parameters from each layer and computing accuracy.
1. Modify *config.ini* on each agent.
  - below is an example of *config.ini* for *layer1*
  ```
 [currentLayer]
layer = 0
 # which layer you want to run on current machine
[common]
total_layers = 7
#total number of layers
iteration = 30
#number of iterations
rho = 1e-4
mu = 0.01
seed_num = 0
neurons = 2000
# number of neurons
plasma_path = /home/ec2-user/plasma
#modify '/home/ec2-user' to an existing path
platform = cpu
#cpu or gpu
chunks = 1
#how many chunks do you want to split the weights
[layer0]
server = 172.31.94.188

[layer1]
server = 172.31.7.13

[layer2]
server = 172.31.13.223

[layer3]
server = 172.31.8.238


  ```
2. On each agent, run the following command. For the detailed use of Plasma, please visit https://arrow.apache.org/docs/python/plasma.html
```
plasma_store -m 30000000000 -s  /home/ec2-user/plasma
```
3. On each agent, run the following command:
```
python3 server_pdADMM.py
```
4. On *layer0*, run:
```
python3 client_pdADMM.py
```

