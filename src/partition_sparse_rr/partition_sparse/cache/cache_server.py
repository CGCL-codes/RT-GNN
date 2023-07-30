import torch
import dgl
import sys
import time
class CacheRedundant:
    def __init__(self,graph,redundant_num,gpuid,cachesize): #cachesize is in MB
        self.graph = graph
        self.redundant_num = redundant_num
        self.gpuid = gpuid

        self.mask = torch.zeros(self.graph.num_nodes()).bool().to('cuda:'+str(gpuid))

        self.cachesize = cachesize *1024*1024 
        self.data_dim = graph.ndata['feat'].size(1)
        self.data_size = sys.getsizeof(graph.ndata['feat'][0][0])
        self.data_row_size = self.data_dim*4
        self.cache_num = self.cachesize//self.data_row_size if self.cachesize//self.data_row_size<self.redundant_num else self.redundant_num

        #cache_data
        self.cpu_part = graph.ndata['feat'][0:(graph.num_nodes()-redundant_num)].clone()
        self.cpu_part = dgl.contrib.UnifiedTensor(self.cpu_part.to('cpu'),device = 'cuda:'+str(gpuid))

        self.gpu_part = graph.ndata['feat'][(graph.num_nodes()-redundant_num):].clone().to('cuda:'+str(gpuid))

        self.mask[(graph.num_nodes()-redundant_num):] = True
    def size(self,dim:int):
        return self.gpu_part.size(dim)
    def __getitem__(self,nid):
        
        nid_gpu = nid.to('cuda:'+str(self.gpuid))
        full_id = torch.zeros(self.graph.num_nodes(),dtype = torch.long,device = 'cuda:'+str(self.gpuid))
        sort_id = torch.arange(0,nid.size(0),device = 'cuda:'+str(self.gpuid)).to(torch.long)
        full_id[nid_gpu] = sort_id

        
        gpu_mask = self.mask[nid_gpu]
        gpu_id = nid_gpu[gpu_mask]
        cpu_mask = ~gpu_mask 
        cpu_id = nid_gpu[cpu_mask]
        gpu_size = gpu_id.shape[0]
        cpu_size = cpu_id.shape[0]
        feat = torch.zeros((nid.size(0),self.data_dim),dtype = torch.float32,device = 'cuda:'+str(self.gpuid))
        cpu_data = self.cpu_part[cpu_id].to('cuda:'+str(self.gpuid))
        gpu_data = self.gpu_part[gpu_id-(self.graph.num_nodes()-self.redundant_num)]
        feat[full_id[cpu_id]] = cpu_data
        feat[full_id[gpu_id]] = gpu_data
        return feat











