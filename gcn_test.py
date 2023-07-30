import dgl
import torch
import numpy as np
import dgl.nn as dglnn

import torch.nn as nn
import torch.nn.functional as F
import time
import argparse

import Fusion as GCNFusion

from gnn_conv import *
from tqdm import *
import matplotlib.pyplot as plt
from partition_sparse import multiprocess_preprocess
import scipy.sparse as sp
from partition_sparse.cache import cache_server

import pandas as pd

BLK_H = 16
BLK_W = 16

def parse_args():
	parser = argparse.ArgumentParser(description='preprocess_test')
	parser.add_argument('--dataset', type=str,default='coauthorP', help='name of dgl dataset')
	parser.add_argument('--epoch', type=int,default=30, help='epochs number')
	parser.add_argument('--embed', type=int,default=128, help='embedding_dim number')
	parser.add_argument('--k', type=int,default=2, help='number of graph patition if --preprocess')
	parser.add_argument('--round', type=int,default=4, help='total number of rounds to perform reduction if --preprocess')
	parser.add_argument('--num_layers', type=int,default=2,help='total number of GNN layers')
	parser.add_argument("--preprocess", action='store_true',default=False,help='Preprocessing or not')
	parser.add_argument("--reorder", action='store_true',default=False,help='Reorder or not')
	
	args = parser.parse_args()
	return args


class GCNNet(torch.nn.Module):
	def __init__(self, input_dim, hidden_dim,output_dim,num_layers):
		super(GCNNet, self).__init__()
		self.layers = nn.ModuleList()
		self.layers.append(GCNConv(input_dim, hidden_dim))
		for i in range(1, num_layers - 1):
			self.layers.append(GCNConv(hidden_dim, hidden_dim))
		self.layers.append(GCNConv(hidden_dim, output_dim))
		self.num_layers = num_layers
	def forward(self, feat, nodePointer, edgeList, adj_coo,block_num):
		h = feat
		for l, (layer) in enumerate(self.layers):
			
			h = layer(h, nodePointer, edgeList, blockPartition, edgeToColumn, edgeToRow, adj_coo,block_num,g_nodes_ts)

			norm =  torch.nn.LayerNorm(h.size(1), eps=1e-5, elementwise_affine=True)
			h = norm(h.to('cpu')).to('cuda:0')

			if l != self.num_layers-1:
				h = F.relu(h)
			
		return h


args = parse_args()


if args.dataset!='IMDB':
	g , redundant_num= multiprocess_preprocess.main(args.dataset,args.k,args.round,args.preprocess,args.reorder)
	print(g)
else :
	adj_temp = sp.load_npz('/home/zjlab/wsy/work1/data/IMDB.npz')
	g_temp = dgl.from_scipy(adj_temp)
	print(g_temp)
	g , redundant_num= multiprocess_preprocess.main(g_temp,args.k,args.round,args.preprocess,args.reorder)
	
g_nodes = g.num_nodes() - redundant_num
g_nodes_ts = torch.zeros(1,dtype = torch.int)
g_nodes_ts[0] = g_nodes


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
adj_csr = g.adj(scipy_fmt='csr')
print(adj_csr.indptr)
nodePointer = torch.IntTensor(adj_csr.indptr)
edgeList = torch.IntTensor(adj_csr.indices)

num_nodes = len(nodePointer) - 1
num_edges = len(edgeList)

num_row_windows = (num_nodes + BLK_H - 1) // BLK_H
edgeToColumn = torch.zeros(num_edges, dtype=torch.int)
edgeToRow = torch.zeros(num_edges, dtype=torch.int)
blockPartition = torch.zeros(num_row_windows, dtype=torch.int)


start = time.perf_counter()
total_block_num = GCNFusion.preprocess(edgeList, nodePointer, num_nodes, BLK_H, BLK_W, blockPartition, edgeToColumn, edgeToRow)
print(blockPartition.shape[0])
block_num = torch.zeros(1,dtype = torch.int)
block_num[0] = total_block_num

print(block_num)
build_neighbor_parts = time.perf_counter() - start
print("Prep. (ms):\t{:.3f}".format(build_neighbor_parts*1e3))


nodePointer = nodePointer.to(device)
edgeList = edgeList.to(device)
blockPartition = blockPartition.to(device)
edgeToColumn = edgeToColumn.to(device)
edgeToRow = edgeToRow.to(device)

n_features = g.ndata['feat'].shape[1]
n_labels = int(g.ndata['label'].max().item() + 1)

model = GCNNet(n_features,args.embed,n_labels,args.num_layers)

if args.preprocess:
	feat = cache_server.CacheRedundant(g,redundant_num,0,100)
	g.ndata.pop('feat')
else:
	feat = g.ndata.pop('feat')


adj_coo = g.adj(scipy_fmt='coo')

if __name__ == "__main__":

	start_train = time.perf_counter()

	opt = torch.optim.Adam(model.parameters())


	for epoch in range(args.epoch):
		
		model.train()
		feats = feat[torch.arange(g.num_nodes())].to(device)
		output_predictions = model(feats, nodePointer, edgeList, adj_coo,block_num)
		loss = F.cross_entropy(output_predictions[:g_nodes],g.ndata['label'][:g_nodes].to('cuda:0'))
		opt.zero_grad()
		loss.backward()
		opt.step()
		print("epoch",epoch," time:",time.perf_counter() - start_train)
	train_time = time.perf_counter() - start_train 

	print("train time:",train_time)

	model.eval()
	predictions = []
	labels = []
	
	with torch.no_grad():
		predictions.append(model(feat[torch.arange(g.num_nodes())].to(device), nodePointer, edgeList, adj_coo,block_num)[:g_nodes].argmax(1).to('cpu').numpy())
		labels.append(g.ndata['label'][:g_nodes].to('cpu').numpy())
		predictions = np.concatenate(predictions)
		labels = np.concatenate(labels)
		result = (predictions==labels).sum().item()/labels.shape[0]
		print(result)
	
	