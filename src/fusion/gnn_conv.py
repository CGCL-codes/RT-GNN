#!/usr/bin/env python3
import torch
import sys
import math
import time 
import numpy as np

from tqdm.std import tqdm
import Fusion as GCNFusion

class GCNFunction(torch.autograd.Function):
	@staticmethod
	def forward(ctx, X, weights, nodePointer, edgeList, blockPartition, edgeToColumn, edgeToRow, adj_coo,block_num,g_nodes):
		
		ctx.save_for_backward(X, weights, nodePointer, edgeList, blockPartition, edgeToColumn, edgeToRow,block_num,g_nodes)
		
		XW = torch.mm(X, weights.to('cuda:0'))
		
		AXW2 = GCNFusion.forward(nodePointer, edgeList, XW, blockPartition, edgeToColumn, edgeToRow,block_num,g_nodes)[0]

		return AXW2

	@staticmethod
	def backward(ctx, d_output):
		X, weights, nodePointer, edgeList, blockPartition, edgeToColumn, edgeToRow ,block_num,g_nodes= ctx.saved_tensors
		
		d_input_prime = GCNFusion.forward(nodePointer, edgeList, d_output ,blockPartition, edgeToColumn, edgeToRow,block_num,g_nodes)[0]


		d_input = torch.mm(d_input_prime, weights.transpose(0,1).to('cuda:0'))
		d_weights = torch.mm(X.transpose(0,1), d_input_prime).to('cuda:0')
		return d_input, d_weights, None, None, None, None, None, None ,None,None


class GCNConv(torch.nn.Module):
	def __init__(self, input_dim, output_dim):
		super(GCNConv, self).__init__()
		
		self.weights = torch.nn.Parameter(torch.Tensor(input_dim, output_dim).to('cuda:0'))
		
		torch.nn.init.xavier_uniform_(self.weights)

	def forward(self, X, nodePointer, edgeList, blockPartition, edgeToColumn, edgeToRow, adj_coo,block_num,g_nodes):
		
		return GCNFunction.apply(X, self.weights, nodePointer, edgeList, blockPartition, edgeToColumn, edgeToRow, adj_coo,block_num,g_nodes)