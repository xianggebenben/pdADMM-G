# pdADMM-G: parallel graph deep learning Alternating Direction Method of Multipliers

This is an implementation of ADMM to achieve communication-efficient model parallelism for the Graph Augmented Multi-Layer Perceptron (GA-MLP) model, as described in our paper:

>Junxiang Wang, Hongyi Li(first-coauthor), Zheng Chai, Yongchao Wang, Yue Cheng, and Liang Zhao. Towards Quantized Model Parallelism for Graph-Augmented MLPs Based on Gradient-Free ADMM Framework
[[Paper]](https://www.researchgate.net/publication/351744585_Towards_Quantized_Model_Parallelism_for_Graph-Augmented_MLPs_Based_on_Gradient-Free_ADMM_framework)


*serial pdADMM-G*  is the source code of serial implementation of the pdADMM-G and pdADMM-G-Q algorithms.

*parallel pdADMM-G* is the source code of parallel implementation of the pdADMM-G and pdADMM-G-Q algorithms.

## Requirements
The codebase is implemented in Python 3.8.10. package versions used for development are just below.
```
torch                1.8.1
torch-cluster        1.5.9
torch-geometric      1.7.1
torch-scatter        2.0.7
torch-sparse         0.6.10
torch-spline-conv    1.2.1
pyarrow              3.0.0
tornado              6.1
```
