#include <torch/extension.h>
#include <vector>
#include <string.h>
#include <cstdlib>
#include <map>

#include <thrust/sort.h>
#include <thrust/execution_policy.h>

#define min(x, y) (((x) < (y))? (x) : (y))

#define CHECK_CUDA(x) TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)



std::vector<torch::Tensor> forward_cuda_V100(
    torch::Tensor nodePointer_tensor,
    torch::Tensor edgeList_tensor,			    
    torch::Tensor XW_tensor,
    torch::Tensor blockPartition_tensor, 
    torch::Tensor edgeToColumn_tensor,
    torch::Tensor edgeToRow_tensor,
    int num_nodes,
    int num_edges,
    int embedding_dim,
    torch::Tensor block_num_tensor,
    torch::Tensor g_nodes_tensor
);


std::vector<torch::Tensor> forward(
    torch::Tensor nodePointer,
    torch::Tensor edgeList,			    
    torch::Tensor XW,
    torch::Tensor blockPartition, 
    torch::Tensor edgeToColumn,
	torch::Tensor edgeToRow,
    torch::Tensor block_num,
    torch::Tensor g_nodes
) {
    CHECK_CUDA(nodePointer);
    CHECK_CUDA(edgeList);
    CHECK_CUDA(XW);
    CHECK_INPUT(blockPartition);
    CHECK_INPUT(edgeToColumn);
    CHECK_INPUT(edgeToRow);

    int num_nodes = nodePointer.size(0) - 1;
    int num_edges = edgeList.size(0);
    int embedding_dim = XW.size(1);

	
    auto res = forward_cuda_V100(nodePointer, edgeList, XW, blockPartition, edgeToColumn, edgeToRow,  num_nodes, num_edges, embedding_dim,block_num,g_nodes);
    return res; 
	
    
}




std::map<unsigned, unsigned> inplace_deduplication(unsigned* array, unsigned length){
    int loc=0, cur=1;
    std::map<unsigned, unsigned> nb2col;
    nb2col[array[0]] = 0;
    while (cur < length){
        if(array[cur] != array[cur - 1]){
            loc++;
            array[loc] = array[cur];
            nb2col[array[cur]] = loc;       
        }
        cur++;
    }
    return nb2col;
}

std::map<unsigned,unsigned> compute_deduplication(unsigned *neighbor_window,unsigned window_size)
{
    std::map<unsigned,unsigned> deduplication;

    for(int i=0;i<window_size;i++)
    {
        if(!deduplication.count(neighbor_window[i])) deduplication[neighbor_window[i]] = 1;
        else deduplication[neighbor_window[i]]++;
    }
    return deduplication;
}

bool cmp(const std::pair<unsigned, unsigned>& a, const std::pair<unsigned, unsigned>& b) {
        return a.second > b.second;
}


unsigned preprocess(torch::Tensor edgeList_tensor, 
                torch::Tensor nodePointer_tensor, 
                int num_nodes, 
                int blockSize_h,
                int blockSize_w,
                torch::Tensor blockPartition_tensor, 
                torch::Tensor edgeToColumn_tensor,
                torch::Tensor edgeToRow_tensor
                ){

    
    // input tensors.
    auto edgeList = edgeList_tensor.accessor<int, 1>();
    auto nodePointer = nodePointer_tensor.accessor<int, 1>();

    // output tensors.
    auto blockPartition = blockPartition_tensor.accessor<int, 1>();
    auto edgeToColumn = edgeToColumn_tensor.accessor<int, 1>();
    auto edgeToRow = edgeToRow_tensor.accessor<int, 1>();

    unsigned block_counter = 0;

    
    #pragma omp parallel for 
    for (unsigned nid = 0; nid < num_nodes; nid++){
        for (unsigned eid = nodePointer[nid]; eid < nodePointer[nid+1]; eid++)
            edgeToRow[eid] = nid;
    }

    unsigned windowId = 0;

    
    #pragma omp parallel for reduction(+:block_counter)
    for (unsigned iter = 0; iter < num_nodes + 1; iter +=  blockSize_h){
        
        unsigned block_start = nodePointer[iter];
        unsigned block_end = nodePointer[min(iter + blockSize_h, num_nodes)];
        unsigned num_window_edges = block_end - block_start;
        if(num_window_edges == 0) continue;
        unsigned *neighbor_window = (unsigned *) malloc (num_window_edges * sizeof(unsigned));
        memcpy(neighbor_window, &edgeList[block_start], num_window_edges * sizeof(unsigned));

        thrust::sort(neighbor_window, neighbor_window + num_window_edges);

        std::map<unsigned,unsigned> deduplication = compute_deduplication(neighbor_window,num_window_edges);

        std::vector<std::pair<unsigned, unsigned>> vec(deduplication.begin(), deduplication.end());

        sort(vec.begin(), vec.end(), cmp);

        blockPartition[windowId] = (vec.size() + blockSize_w - 1) /blockSize_w;

        block_counter += blockPartition[windowId];

        windowId++;

        std::map<unsigned,unsigned> edge_pos;

        for(int i=0;i<vec.size();i++)
        {
            edge_pos.insert(std::make_pair(vec[i].first,i));
        }

        for (unsigned e_index = block_start; e_index < block_end; e_index++){
            unsigned eid = edgeList[e_index];
            edgeToColumn[e_index] = edge_pos[eid];
        }

        
    }
    printf("TC_Blocks:\t%d\nExp_Edges:\t%d\n", block_counter, block_counter * 16 * 16);
    return windowId;

    
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("preprocess", &preprocess, "Preprocess Step (CPU)");

    m.def("forward", &forward, "forward function (CUDA)");
    m.def("backward", &forward, "backward function (CUDA)");
}