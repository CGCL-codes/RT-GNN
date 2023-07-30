# RT-GNN: Accelerating Sparse Graph Neural Networks by Tensor-CUDA Kernel Fusion

RT-GNN is a block-based row-wise multiplication approach with redundancy elimination. Firstly, we construct HEG, a novel graph representation that eliminates redundancy in GNNs through a greedy search. Secondly, to improve data locality and reuse during the training and inference, we propose block-based row-wise multiplication to alleviate memory constraints and benefit from advanced hardware. Finally, we propose Fuser to parallelize TC and CD by fusing kernels based on different computation paradigms in GNNs.

## Introduction

Graph Neural Networks (GNNs) have achieved remarkable successes in various graph learning tasks, thanks to their ability to leverage advanced GPUs. However, GNNs currently face challenges with concurrent use of advanced Tensor Cores (TCs) and CUDA Cores (CDs) in GPUs. These challenges arise due to repeated redundant computations and inefficient data locality that result from the large size, high sparsity, and irregular non-zero distribution of real-world graphs.

We propose RT-GNN, a novel framework for accelerating GNNs through the fusion of advanced TC and CD cores. A hierarchical embedding graph (HEG) is introduced to efficiently eliminates redundant computation. By characterizing the redundancy through the use of adjacency matrix properties, HEG outperforms traditional approaches obviously. We also present a block-based row-wise multiplication method that addresses the inherent sparsity of graphs. HEG is divided into small blocks (a.k.a tiles) to make better use of the CD hardware and maximize reuse of feature and result matrices. Further, an adaptive architecture Fuser is proposed to deploy different types of tiles from HEG into TC and CD units according to their sparsity after a lightweight graph reordering, allowing efficient parallel processing. Experimental results demonstrate that RT-GNN outperforms HAG by an average of 19.25×, with a remarkable 72× improvement on the ARXIV dataset for redundancy elimination. Moreover, for overall performance RT-GNN outperforms state-of-the-art GNN computing frameworks by an average of 3.31×, with a notable 3.57× improvement in comparison to TC-GNN.

## HEG Representation

<div align=center>
<img src="https://github.com/CGCL-codes/RT-GNN/blob/main/imgs/HEG.png" width="450" height="260" alt="HEG Representation"/><br/>
</div>


To effectively reduce redundant computing, RT-GNN introduces redundancy elimination technique to construct HEG by adjacency matrix properties, aiming to hierarchically managing and reusing intermediate aggregation results. Graph in the real world may follow a power-law distribution. Nodes within a partition are highly likely to share common neighbors.

## Block-based Row-wise Multiplication

<div align=center>
<img src="https://github.com/CGCL-codes/RT-GNN/blob/main/imgs/Fuser.png" width="500" height="280" alt="Block-based Row-wise Multiplication"/><br/>
</div>

The irregular aggregation computation for SpMM lead to high memory usage, and low data reuse.So we propose block-based row-wise multiplication approach and Fuser to compute GNN models, in order to exploit the graph data locality and cores execution opportunity. Fuser dynamically combines the Tensor and CUDA Cores. The fundamental building block in our proposal is graph tile, which divides a graph into smaller blocks, a.k.a., tiles.

## About the source code and data sets

We have implemented the basic idea HEG(parallel_sparse_rr.py) in python,Block-based Row-wise Multiplication(GCNFusion_V100.cu) in cuda.

All datasets can be downloaded through dgl and ogb,you can adjust the required dataset through the dataset parameters and automatically download it.

## Environment

Our code runs on Linux 18.04, with a hardware environment of CPU: Intel Xeon Processor (Skylake, IBRS), and GPU: NVIDIA Tesla V100.

Python>=3.8

dgl>=0.8

pytorch>=1.10

cuda>=10

## How to run

### Run the demo

Suppose you've already cloned the repository.  
You just need:

```
    $ cd src/fusion && python setup.py develop
    $ cd src/partition_sparse_rr && python setup.py develop 
```

### Parameter setting

```
   --dataset:name of dgl dataset
   --epoch:epochs number
   --embed:embedding_dim number
   --k:number of graph patition if --preprocess
   --round:total number of rounds to perform reduction if --preprocess
   --num_layers:total number of GNN layers
   --preprocess:Preprocessing or not
```

## 
