import time

#import csrgemm
import numpy as np
import scipy.sparse as sp
import torch


def construct_ga(adj_gs):

    num_v_sub = adj_gs.shape[0]

    adj_gs_T = adj_gs.T
    
    redundancy_matrix = adj_gs_T.dot(adj_gs)
    
    redundancy_coo = redundancy_matrix.tocoo()

    weight_edges = np.stack((redundancy_coo.row,redundancy_coo.col,redundancy_coo.data)).T
    print(weight_edges.shape)
    return weight_edges,num_v_sub,redundancy_matrix

def find_node_max_redundancy(indices,value,vis,num,node):
    max = 0
    maxid = -1
    for i in range(num):
        if vis[indices[i]] == 1:
            continue
        if indices[i] == node :
            continue
        if value[i] > max and value[i]>2:
            max = value[i]
            maxid = indices[i]     
    return (node,maxid)

def find_max_redundancy(redundancy_matrix,node_id,num_v_sub):
    
    indptr = redundancy_matrix.indptr
    indices = redundancy_matrix.indices
    redundancy_num = redundancy_matrix.data
    vis = np.zeros(num_v_sub)
    M = []
    for i in range(num_v_sub):
        if vis[i]==1 :
            continue
        if indptr[i+1]-indptr[i] == 0 :
            continue
        indices_i = indices[indptr[i]:indptr[i+1]]
        value_i = redundancy_num[indptr[i]:indptr[i+1]]
        res = find_node_max_redundancy(indices_i,value_i,vis,indptr[i+1]-indptr[i],i)
        if res[1] != -1:
            vis[res[0]] = 1
            vis[res[1]] = 1
            M.append((node_id[res[0]],node_id[res[1]]))
        if len(M) == int(num_v_sub/2):
            break
    return M
        
    
    
def obtain_precompute_edges(weight_edges,node_id,num_v_sub):


    
    M = []
    data = weight_edges.T[2]
    H = weight_edges[data>2]
    
    
    H_sorted = sorted(H,key = lambda x:(x[2]),reverse=True)
    
    
    
    S = np.ones(num_v_sub)
    _W = 0
    for row,col,data in H_sorted:
        if not(S[row] and S[col]):
            continue
        if row == col:
            continue
        _W += data -1
        S[row] = 0
        S[col] = 0
        M.append((node_id[row],node_id[col]))
        if len(M) == int(num_v_sub/2):
            break
    return M,_W

def obtain_compact_mat(adj_gs,adj_t,M,feat):
    
    gs_t_indptr= adj_t.indptr
    gs_t_indices= adj_t.indices

    agg_pair = list()
    agg_res = list()
    agg_num = 0

    for (aggr1,aggr2) in M:
        # intersection of aggr1's neighbor and aggr2's neighbor
        _neigh1 = gs_t_indices[gs_t_indptr[aggr1]:gs_t_indptr[aggr1+1]]
        _neigh2 = gs_t_indices[gs_t_indptr[aggr2]:gs_t_indptr[aggr2+1]]
        agg_pair.append((aggr1,aggr2))
        agg_res.append(np.intersect1d(_neigh1,_neigh2,assume_unique=True))
        agg_num +=1
    return agg_pair,agg_res,agg_num
    

f_tot_ops = lambda adj: adj.size-np.where(np.ediff1d(adj.indptr)>0)[0].size
f_tot_read = lambda adj: adj.size
max_deg = lambda adj: np.ediff1d(adj.indptr).max()
mean_deg = lambda adj: np.ediff1d(adj.indptr).mean()
sigma_deg2 = lambda adj: (np.ediff1d(adj.indptr)**2).sum()/adj.shape[0]

def main(adj_sub,adj,adj_t,node_id,feat):
    adj_gs = adj_sub
    
    weight_edges,num_v_sub,mat= construct_ga(adj_gs)
    
    M = find_max_redundancy(mat,node_id,num_v_sub)  
    
    agg_pair,agg_res,agg_num = obtain_compact_mat(adj,adj_t,M,feat)
    
    return agg_pair,agg_res,agg_num
