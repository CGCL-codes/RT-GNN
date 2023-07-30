import argparse
import os.path
import time
from multiprocessing import shared_memory
from ogb.graphproppred import DglGraphPropPredDataset

import dgl
import numpy as np
import scipy.sparse as sp
import torch
from dgl.partition import metis_partition_assignment
from partition_sparse.data.rr import parallel_sparse_rr
from ogb.nodeproppred import DglNodePropPredDataset

from torch import multiprocessing



def rr(node_part,part_offset,g_new,k,index,p,q,max_num,agg_pair,agg_res,agg_num):
    num = p+1 if index<q else p
    for i in range(num):
        id = i*max_num + index
        node_id = node_part[part_offset[id]:part_offset[id+1]]
        g_sub = g_new.subgraph(node_id)
        adj_sub = g_sub.adj(scipy_fmt='csr')
        agg_pair[id],agg_res[id],agg_num[id] = parallel_sparse_rr.main(adj_sub,g_new.adj(scipy_fmt='csr'),g_new.adj(scipy_fmt='csr',transpose=True),node_id.numpy(),g_new.ndata['feat'].numpy())

def reconstruction_graph(index,p,q,max_num,agg_pair,agg_res,agg_num,indptr,
                          feat_name,deg_reduce_name,indices_name,
                          feat_shape,deg_reduce_shape,indices_shape,
                          feat_type,deg_reduce_type,indices_type,):
    num = p+1 if index<q else p
    offset_arr = np.zeros(len(agg_num)+1,dtype = np.int32)
    offset_arr[1:] = np.cumsum(agg_num)

    feat_shm = shared_memory.SharedMemory(name=feat_name)
    indices_shm = shared_memory.SharedMemory(name=indices_name)
    deg_reduce_shm = shared_memory.SharedMemory(name=deg_reduce_name)

    feat = np.ndarray(feat_shape, dtype=feat_type, buffer=feat_shm.buf)
    indices = np.ndarray(indices_shape, dtype=indices_type, buffer=indices_shm.buf)
    deg_reduce = np.ndarray(deg_reduce_shape, dtype=deg_reduce_type, buffer=deg_reduce_shm.buf)

    for i in range(num):
        id = i*max_num + index
        pair = agg_pair[id]
        res = agg_res[id]                                                    
        num_pair = agg_num[id]

        offset = offset_arr[id]

        idx = 0
        num_v = deg_reduce.shape[1]
        for j in range(num_pair):
            (agg1,agg2) = pair[j]
            root = res[j]
            feat[num_v+offset+idx] = feat[agg1]+feat[agg2]
            for node in root:
                neigh = indices[indptr[node]:indptr[node+1]]
                i1 = np.where(neigh==agg1)[0][0]
                i2 = np.where(neigh==agg2)[0][0]       
                indices[indptr[node]+i1] = num_v+offset+idx
                indices[indptr[node]+i2] = -1
                deg_reduce[id][node] +=1
            idx+=1






def parallel(node_part,part_offset,g_new,k):
    n_cpu = multiprocessing.cpu_count()
    p = int(k / n_cpu)
    q = int(k % n_cpu)
    max_num = n_cpu if k>n_cpu else k
    process_list = []
    manager = multiprocessing.Manager()
    agg_pair = manager.list()
    agg_res = manager.list()
    agg_num = manager.list()
    for i in range(k):
        agg_pair.append(manager.list())
        agg_res.append(manager.list())
        agg_num.append(0)
    for i in range(max_num):
        process = multiprocessing.Process(target = rr,args = (node_part,part_offset,g_new,k,i,p,q,max_num,agg_pair,agg_res,agg_num))
        process.start()
        process_list.append(process)
    for process in process_list:
        process.join()
    
    total_num = 0
    for i in range(k):
        total_num+=agg_num[i]
        agg_pair[i] = list(agg_pair[i])
        agg_res[i] = list(agg_res[i])
    agg_pair = list(agg_pair)
    agg_res = list(agg_res)


    feat = g_new.ndata['feat'].numpy()
    adj = g_new.adj(scipy_fmt='csr')
    new_feat = np.zeros((feat.shape[0]+total_num,feat.shape[1]))
    new_feat[:feat.shape[0]] = feat.copy()

    deg = np.ediff1d(adj.indptr).astype(np.int32)

    deg_reduce = np.zeros((k,g_new.num_nodes()),dtype = np.int32)


    num_v = feat.shape[0]

    feat_shm = shared_memory.SharedMemory(create=True, size=new_feat.nbytes)
    indices_shm = shared_memory.SharedMemory(create=True, size=adj.indices.nbytes)
    deg_reduce_shm = shared_memory.SharedMemory(create=True, size=deg_reduce.nbytes)

    feat_shm_arr = np.ndarray(new_feat.shape,dtype = new_feat.dtype,buffer = feat_shm.buf)
    indices_shm_arr = np.ndarray(adj.indices.shape,dtype = adj.indices.dtype,buffer = indices_shm.buf )
    deg_reduce_shm_arr = np.ndarray(deg_reduce.shape,dtype = deg_reduce.dtype,buffer = deg_reduce_shm.buf)

    feat_shm_arr[:] = new_feat[:]
    
    indices_shm_arr[:] = adj.indices[:]
    deg_reduce_shm_arr[:] = deg_reduce[:]



    process_list = []
    for i in range(max_num):
        process = multiprocessing.Process(target = reconstruction_graph,args = (i,p,q,max_num,agg_pair,agg_res,agg_num,adj.indptr,
                                                                                    feat_shm.name,deg_reduce_shm.name,indices_shm.name,
                                                                                    new_feat.shape,deg_reduce.shape,adj.indices.shape,
                                                                                    new_feat.dtype,deg_reduce.dtype,adj.indices.dtype))
        process.start()
        process_list.append(process)
    for process in process_list:
        process.join()


    reduce_total = np.sum(deg_reduce_shm_arr,axis = 0)

    _deg = deg - reduce_total

    _indptr_new = np.cumsum(_deg)
    indptr_new = np.zeros(feat_shm_arr.shape[0]+1,dtype = np.int32)
    indptr_new[1:num_v+1] = _indptr_new
    indptr_new[num_v+1:] = _indptr_new[-1]
    indices_new = indices_shm_arr[np.where(indices_shm_arr>-1)]


    print(indptr_new[-1])
    print(indices_new.size)
    assert indices_new.size == indptr_new[-1]
    data_new = np.ones(indices_new.size)
    new_adj = sp.csr_matrix((data_new,indices_new,indptr_new),shape=(feat_shm_arr.shape[0],feat_shm_arr.shape[0]))

    feat_shm.close()
    feat_shm.unlink()

    deg_reduce_shm.close()
    deg_reduce_shm.unlink()

    indices_shm.close()
    indices_shm.unlink()


    return new_adj,new_feat
    




def main(data,k,round,pre_flag,reorder_flag):
    if data == 'cite':
        dataset = dgl.data.CiteseerGraphDataset()
    elif data=='kara':
        dataset = dgl.data.KarateClubDataset()
    elif data =='cora':
        dataset = dgl.data.CoraFullDataset()
    elif data=='ppi':
        dataset = dgl.data.PPIDataset()
    elif data =='coauthorP':
        dataset = dgl.data.CoauthorPhysicsDataset()
    elif data =='coauthorC':
        dataset = dgl.data.CoauthorCSDataset()
    elif data =='pubmed':
        dataset = dgl.data.PubmedGraphDataset()
    elif data =='amazon':
        dataset = dgl.data.AmazonCoBuyComputerDataset()
    elif data =='reddit':
        dataset = dgl.data.RedditDataset()
    elif data =='wiki':
        dataset = dgl.data.WikiCSDataset()
    elif data =='flickr':
        dataset = dgl.data.FlickrDataset()
    elif data =='yelp':
        dataset = dgl.data.YelpDataset()
    elif data =='products':
        dataset = DglNodePropPredDataset(name='ogbn-products',root = '/home/hda/ogb_dataset')
    elif data =='arxiv':
        dataset = DglNodePropPredDataset(name='ogbn-arxiv',root = '/home/zjlab/hda/ogb_dataset')
    else:
        dataset = data
    
    print('Dataset loading completed')
    if type(data)!=type(""):
        g = dataset
        g.ndata['feat'] =torch.zeros((g.num_nodes(),1024),dtype = torch.float32)
        g.ndata['label'] = torch.zeros(g.num_nodes(),dtype = torch.int64)
        data = 'IMDB'
    elif data =='products' or data == 'arxiv':
        g,label = dataset[0]
        g.ndata['label'] = label.squeeze(dim = 1)
    elif data=='kara':
        g = dataset[0]
        g.ndata['feat'] = torch.zeros((g.num_nodes(),10),dtype = torch.float32)
    else:
        g = dataset[0]
    #split train,valid,test
    nids = np.arange(g.num_nodes())
    np.random.shuffle(nids)
    train_len = int(g.num_nodes()*0.6)
    val_len = int(g.num_nodes()*0.2)

    #not preprocess
    if not pre_flag:
        g = dgl.add_self_loop(g)
        # train mask
        train_mask = np.zeros(g.num_nodes(), dtype=np.int)
        train_mask[nids[0:train_len]] = 1
        g.ndata['train_mask'] = torch.from_numpy(train_mask).to(torch.bool)

        # val mask
        val_mask = np.zeros(g.num_nodes(), dtype=np.int)
        val_mask[nids[train_len:train_len + val_len]] = 1
        g.ndata['val_mask'] = torch.from_numpy(val_mask).to(torch.bool)

        # test mask
        test_mask = np.zeros(g.num_nodes(), dtype=np.int)
        test_mask[nids[train_len + val_len:g.num_nodes()]] = 1
        g.ndata['test_mask'] = torch.from_numpy(test_mask).to(torch.bool)
        if reorder_flag:
            g = dgl.reorder_graph(g, node_permute_algo='rcmk')
            return g,0
        else:
            return g,0
    

    labels = g.ndata['label']
    out_dir = '/home/zjlab/hda/partition_sparse_rr/partition_sparse/data/dataset'
    
    adj_file_name = data+'_adj_'+str(round)+'.npz'
    feat_file_name = data+'_feat_'+str(round)+'.npy'
    
    adj_file = os.path.join(out_dir,adj_file_name)
    feat_file = os.path.join(out_dir,feat_file_name)

    part_time = 0
    
    
    g_new = g
    if os.path.isfile(adj_file) and os.path.isfile(feat_file):
        adj_new = sp.load_npz(os.path.join(out_dir,adj_file_name))
        feats_new = np.load(os.path.join(out_dir,feat_file_name))
        g_new = dgl.from_scipy(adj_new)
        g_new.ndata['feat'] = torch.from_numpy(feats_new).to(torch.float32)
    else:
        time1 = time.time()
        start_round = 0
        for base in range(1,round):
            adj_base = os.path.join(out_dir,data+'_adj_'+str(round-base)+'.npz')
            feat_base = os.path.join(out_dir,data+'_feat_'+str(round-base)+'.npy')
            if os.path.isfile(adj_base) and os.path.isfile(feat_base):
                adj_new = sp.load_npz(adj_base)
                feats_new = np.load(feat_base)
                g_new = dgl.from_scipy(adj_new)
                g_new.ndata['feat'] = torch.from_numpy(feats_new).to(torch.float32)
                start_round = round - base
                break
        
        
        for r in range(start_round,round):
            print(g_new)
            print(r+1,"round of redundancy elimination")
            time_round1 = time.time()
            node_part = metis_partition_assignment(g_new,k)#,balance_edges=True)
            part_time = part_time + time.time()-time_round1
            
            part_num = torch.zeros(k)
            for i in range(g_new.num_nodes()):
                part_num[node_part[i]] +=1
            part_offset = torch.cumsum(part_num,dim=0)
            part_offset = torch.cat([torch.zeros(1),part_offset],dim=0).to(torch.int32)
            node_part = torch.argsort(node_part)
            
            g_temp = g_new.subgraph(node_part)
            new_adj,new_feat = parallel(node_part,part_offset,g_new,k)
            
            g_new = dgl.from_scipy(new_adj)
            g_new.ndata['feat'] = torch.from_numpy(new_feat).to(torch.float32)

            time_round2= time.time()
            print(r+1,"round of redundancy elimination time:",time_round2-time_round1)
            
            #save data
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)
            sp.save_npz(os.path.join(out_dir,data+'_adj_'+str(r+1)+'.npz'),g_new.adj(scipy_fmt='csr'))
            np.save(os.path.join(out_dir,data+'_feat_'+str(r+1)+'.npy'),g_new.ndata['feat'].numpy()) 
              
        time2 = time.time()
        print("redundancy elimination time:",time2-time1)  
    

    n_labels = int(labels.max().item() + 1)
    labels_new = torch.ones(g_new.num_nodes())*(n_labels)
    labels_new[:g.num_nodes()] = labels
    g_new.ndata['label'] = labels_new.to(torch.int64)
    
    # train mask
    train_mask = np.zeros(g_new.num_nodes(), dtype=np.int)
    train_mask[nids[0:train_len]] = 1
    g_new.ndata['train_mask'] = torch.from_numpy(train_mask).to(torch.bool)

    # val mask
    val_mask = np.zeros(g_new.num_nodes(), dtype=np.int)
    val_mask[nids[train_len:train_len + val_len]] = 1
    g_new.ndata['val_mask'] = torch.from_numpy(val_mask).to(torch.bool)

    # test mask
    test_mask = np.zeros(g_new.num_nodes(), dtype=np.int)
    test_mask[nids[train_len + val_len:g.num_nodes()]] = 1
    g_new.ndata['test_mask'] = torch.from_numpy(test_mask).to(torch.bool)

    if reorder_flag:
        g_new = dgl.reorder_graph(g_new, node_permute_algo='rcmk')

    return g_new,g_new.num_nodes()-g.num_nodes()



