#include <torch/extension.h>
#include <stdio.h>
#include <vector>

#include <thrust/device_vector.h>
#include <thrust/sort.h>

#include <cuda.h>
#include <mma.h>
#include <cuda_runtime.h>
// #include <cuda_fp16.h>

// #define SM_NUM 108 // A100
#define SM_NUM 80 // V100
#define WARP_SIZE 32

#define BLK_H 16
// #define BLK_W 8
#define BLK_W 16

// #define WARP_PER_BLOCK 24

using namespace nvcuda;

#define CHECK_CUDA(func)                                                       \
{                                                                              \
    cudaError_t status = (func);                                               \
    if (status != cudaSuccess) {                                               \
        printf("CUDA API failed at line %d with error: %s (%d)\n",             \
               __LINE__, cudaGetErrorString(status), status);                  \
                                                           \
    }                                                                          \
}

#define CHECK_CUSPARSE(func)                                                   \
{                                                                              \
    cusparseStatus_t status = (func);                                          \
    if (status != CUSPARSE_STATUS_SUCCESS) {                                   \
        printf("CUSPARSE API failed at line %d with error: %s (%d)\n",         \
               __LINE__, cusparseGetErrorString(status), status);              \
                                                           \
    }                                                                          \
}


__global__ void mix_kernel_last_embedding(
	const int * __restrict__ nodePointer,
  	const int *__restrict__ edgeList,
	const int *__restrict__ blockPartition,
	const int *__restrict__ edgeToColumn,
	const int *__restrict__ edgeToRow,
 	const int num_nodes,
	const int num_edges,
	const int embedding_dim,
	const int total_block_num,
	const int tile_num_per_XW_line,
  	const float *__restrict__ XW,
  	float *output	
);

__device__ __forceinline__ void namedBarrierSync(int name, int numThreads);


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
 ) {

	
	auto edgeList = edgeList_tensor.data<int>();
    auto nodePointer = nodePointer_tensor.data<int>();
	auto edgeToRow = edgeToRow_tensor.data<int>();
	auto XW = XW_tensor.data<float>();
	auto blockPartition = blockPartition_tensor.data<int>();
	auto edgeToColumn = edgeToColumn_tensor.data<int>();
	auto block_num = block_num_tensor.data<int>();
	auto g_nodes = g_nodes_tensor.data<int>();

    cudaEvent_t startKERNEL;
    cudaEvent_t stopKERNEL;
    CHECK_CUDA( cudaEventCreate(&startKERNEL) )
    CHECK_CUDA( cudaEventCreate(&stopKERNEL) )

	
    int mix_blocks = 1;
    const int mix_grid_dimension_x = mix_blocks * SM_NUM;
	const int warp_num = 32;
    
    auto output = torch::zeros_like(XW_tensor);
	

	const int total_block_num = block_num[0];
	//auto output = torch::zeros({g_nodes[0], embedding_dim},torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA, 0));

    const int tile_num_per_XW_line = (embedding_dim + BLK_H - 1) / BLK_H;

	const int dense_tc_XW_dynamic_shared_size = 3 * (BLK_W * BLK_H * tile_num_per_XW_line + 8) * sizeof(half);
	const int output_cd_kernel_dynamic_shared_size = (BLK_H * (BLK_H * tile_num_per_XW_line + 1) + 4) * sizeof(float);
	const int output_tc_kernel_dynamic_shared_size = 2 * (BLK_H * BLK_H * tile_num_per_XW_line + 4) * sizeof(float);
	const int dense_cd_XW_dynamic_shared_size = (BLK_W * BLK_H * tile_num_per_XW_line + 4) * sizeof(float);
	const int dynamic_shared_size = dense_cd_XW_dynamic_shared_size + 
										dense_tc_XW_dynamic_shared_size +
										output_cd_kernel_dynamic_shared_size +
										output_tc_kernel_dynamic_shared_size;

	dim3 mix_grid(mix_grid_dimension_x*2, 1, 1);
	dim3 mix_block(WARP_SIZE, warp_num, 1);
	CHECK_CUDA( cudaEventRecord(startKERNEL) )

	// printf("mix_kernel_last_embedding begin\n");
	mix_kernel_last_embedding<<<mix_grid, mix_block, dynamic_shared_size>>>(
						nodePointer, 
						edgeList,
						blockPartition,
						edgeToColumn,
						edgeToRow,
						num_nodes,
						num_edges,
						embedding_dim,
						total_block_num,
						tile_num_per_XW_line,
						XW, 
						output.data<float>());
	cudaDeviceSynchronize();
	// printf("mix_kernel_last_embedding accomplish\n");

	CHECK_CUDA( cudaEventRecord(stopKERNEL) )
	
	CHECK_CUDA(cudaEventSynchronize(stopKERNEL))

	float time;
 
	cudaEventElapsedTime(&time, startKERNEL, stopKERNEL);

	cudaError_t error = cudaGetLastError();
	
    if(error != cudaSuccess)
    {
        // print the CUDA error message and exit
        printf("CUDA error: %s\n", cudaGetErrorString(error));
        exit(-1);
    }
	
    return {output};
}

__launch_bounds__(WARP_SIZE * 32, 2)
__global__ void mix_kernel_last_embedding(
	const int * __restrict__ nodePointer,
  	const int *__restrict__ edgeList,
	const int *__restrict__ blockPartition,
	const int *__restrict__ edgeToColumn,
	const int *__restrict__ edgeToRow,
 	const int num_nodes,
	const int num_edges,
	const int embedding_dim,
	const int total_block_num,
	const int tile_num_per_XW_line,
  	const float *__restrict__ XW,
  	float *output	
) {

	unsigned bid = blockIdx.x;								        // block_index
	const unsigned wid = threadIdx.y;								// warp_index
	const unsigned laneid = threadIdx.x;							// lanid of each warp
	const unsigned tid = threadIdx.y * blockDim.x + laneid;			// threadid of each block
	const unsigned thread_per_block = blockDim.x * blockDim.y;		// number of threads per block
	const unsigned dense_bound = num_nodes * embedding_dim;

	unsigned row_nIdx_start = 0;
	unsigned row_nIdx_end = 0;
	// unsigned cd_start = 0;
	unsigned col_nIdx_tc_start = 0;
	unsigned col_nIdx_tc_end_one = 0;
	unsigned col_nIdx_tc_end_two = 0;
	unsigned col_nIdx_tc_end_three = 0;
	unsigned col_nIdx_cd_start = 0;
	unsigned col_nIdx_cd_end = 0;
	unsigned eIdx_start = 0;
	unsigned eIdx_end = 0;

	unsigned row_local = 0;
	unsigned col_local = 0;

	unsigned tile_num_per_block = 0; // SGT之后一个block中有多少个tile
	unsigned cir_times = 0;

	unsigned dense_rowIdx = 0;
	unsigned dense_dimIdx = 0;
	unsigned source_idx = 0;
	unsigned target_idx = 0;


	
  	__shared__ float sparse_cd_A[BLK_H * (BLK_W + 1)];
	__shared__ unsigned sparse_A_cd_ToXW_index[BLK_W];

	__shared__ half sparse_tc_A_one[BLK_H * BLK_W];
	__shared__ unsigned sparse_A_tc_ToXW_index_one[BLK_W];
	__shared__ half sparse_tc_A_two[BLK_H * BLK_W];
	__shared__ unsigned sparse_A_tc_ToXW_index_two[BLK_W];
	__shared__ half sparse_tc_A_three[BLK_H * BLK_W];
	__shared__ unsigned sparse_A_tc_ToXW_index_three[BLK_W];
	
	extern __shared__ float templist[];

	float *output_cd_kernel = templist;
	float *output_tc_kernel_one = &output_cd_kernel[BLK_H * (BLK_H * tile_num_per_XW_line + 1) + 4];
	float *output_tc_kernel_two = &output_tc_kernel_one[BLK_H * BLK_H * tile_num_per_XW_line + 4];
	
	float *dense_cd_XW = &output_tc_kernel_two[BLK_H * BLK_H * tile_num_per_XW_line + 4];
	half *dense_tc_XW_one = (half*)&dense_cd_XW[BLK_W * BLK_H * tile_num_per_XW_line + 4];
	half *dense_tc_XW_two = (half*)&dense_tc_XW_one[BLK_W * BLK_H * tile_num_per_XW_line + 8];
	half *dense_tc_XW_three = (half*)&dense_tc_XW_two[BLK_W * BLK_H * tile_num_per_XW_line + 8];

	float *output_tc_kernel_three = dense_cd_XW;

	for (;; bid += gridDim.x) {
		if (bid >= total_block_num) {
			// printf("%d ", bid);
			__syncthreads();
            return;
        }

		wmma::fragment<wmma::matrix_a, BLK_H, BLK_H, BLK_W, half, wmma::row_major> a_frag;
		wmma::fragment<wmma::matrix_b, BLK_H, BLK_H, BLK_W, half, wmma::col_major> b_frag;
		wmma::fragment<wmma::accumulator, BLK_H, BLK_H, BLK_W, float> acc_frag;
		wmma::fill_fragment(acc_frag, 0.0f);

		col_nIdx_tc_end_three = 0;

		// 由于是持久线程块，所以必须在for循环里面定义
		tile_num_per_block= blockPartition[bid];
		// cd_start = (tile_num_per_block + 1) / 2; 
		row_nIdx_start = bid * BLK_H;
		row_nIdx_end = min(row_nIdx_start + BLK_H, num_nodes);
		eIdx_start = nodePointer[row_nIdx_start];
		eIdx_end = nodePointer[row_nIdx_end];

		#pragma unroll
		for (unsigned idx = tid; idx < BLK_H * BLK_H * tile_num_per_XW_line + 4; idx += thread_per_block) {
			output_tc_kernel_one[idx] = 0.0f;
			output_tc_kernel_two[idx] = 0.0f;
			output_tc_kernel_three[idx] = 0.0f;
		}

		#pragma unroll
		for (unsigned idx = tid; idx < BLK_H * (BLK_H * tile_num_per_XW_line + 1) + 4; idx += thread_per_block) {
			output_cd_kernel[idx] = 0.0f;
		}

		cir_times = tile_num_per_block / 4 + 1;
		for (unsigned i = 0; i < cir_times; i++) {
			if (4 * (i + 1) <= tile_num_per_block) {
				// if (tid == 0) printf("bid: %d i: %d, ",bid, i);
				col_nIdx_tc_start = col_nIdx_tc_end_three;
				col_nIdx_tc_end_one = col_nIdx_tc_start + BLK_W;
				col_nIdx_tc_end_two = col_nIdx_tc_end_one + BLK_W;
				col_nIdx_tc_end_three = col_nIdx_tc_end_two + BLK_W;

				col_nIdx_cd_start = (tile_num_per_block - i - 1) * BLK_W;
				col_nIdx_cd_end = (tile_num_per_block - i) * BLK_W;
		
				if (tid < BLK_W){
					sparse_A_cd_ToXW_index[tid] = num_nodes + 1;
					sparse_A_tc_ToXW_index_one[tid] = num_nodes + 1;
					sparse_A_tc_ToXW_index_two[tid] = num_nodes + 1;
					sparse_A_tc_ToXW_index_three[tid] = num_nodes + 1;
				}

				#pragma unroll
				for (unsigned idx = tid; idx < BLK_W * BLK_H; idx += thread_per_block) {
					sparse_tc_A_one[idx] = __float2half(0.0f);
					sparse_tc_A_two[idx] = __float2half(0.0f);
					sparse_tc_A_three[idx] = __float2half(0.0f);
				}

				#pragma unroll
				for (unsigned idx = tid; idx < (BLK_W + 1) * BLK_H; idx += thread_per_block) {
					sparse_cd_A[idx] = 0.0f;
				}

				#pragma unroll
				for (unsigned idx = tid; idx < BLK_W * BLK_H * tile_num_per_XW_line + 8; idx += thread_per_block){
					dense_tc_XW_one[idx] = __float2half(0.0f);
					dense_tc_XW_two[idx] = __float2half(0.0f);
					dense_tc_XW_three[idx] = __float2half(0.0f);
				}

				#pragma unroll
				for (unsigned idx = tid; idx < BLK_W * BLK_H * tile_num_per_XW_line + 4; idx += thread_per_block){
					dense_cd_XW[idx] = 0.0f;
				}

				__syncthreads();

				#pragma unroll
				for (unsigned eidx = eIdx_start + tid; eidx < eIdx_end; eidx += thread_per_block) {
					unsigned col = edgeToColumn[eidx];
					
					if (col >= col_nIdx_tc_start && col < col_nIdx_tc_end_one) {
						row_local = edgeToRow[eidx] % BLK_H;
						col_local = col % BLK_W;
						sparse_tc_A_one[row_local * BLK_W + col_local] = __float2half(1.0f);
						sparse_A_tc_ToXW_index_one[col_local] = edgeList[eidx];
					} 
					else if (col >= col_nIdx_tc_end_one && col < col_nIdx_tc_end_two) {
						row_local = edgeToRow[eidx] % BLK_H;
						col_local = col % BLK_W;
						sparse_tc_A_two[row_local * BLK_W + col_local] = __float2half(1.0f);
						sparse_A_tc_ToXW_index_two[col_local] = edgeList[eidx];
					} 
					else if (col >= col_nIdx_tc_end_two && col < col_nIdx_tc_end_three) {
						row_local = edgeToRow[eidx] % BLK_H;
						col_local = col % BLK_W;
						sparse_tc_A_three[row_local * BLK_W + col_local] = __float2half(1.0f);
						sparse_A_tc_ToXW_index_three[col_local] = edgeList[eidx];
					} 
					else if (col >= col_nIdx_cd_start && col < col_nIdx_cd_end) {
						row_local = edgeToRow[eidx] % BLK_H;
						col_local = col % BLK_W;
						sparse_cd_A[row_local * (BLK_W + 1) + col_local] = 1.0f;
						sparse_A_cd_ToXW_index[col_local] = edgeList[eidx];
					}
				}

				__syncthreads();

				switch (wid / tile_num_per_XW_line) {
					case 0: {
						#pragma unroll
						for (unsigned idx = laneid; idx < BLK_W * BLK_H; idx += WARP_SIZE){
							dense_rowIdx = sparse_A_tc_ToXW_index_one[idx % BLK_W];
							dense_dimIdx = idx / BLK_W;
							source_idx = dense_rowIdx * embedding_dim + wid * BLK_H + dense_dimIdx;
							target_idx = wid * BLK_W * BLK_H + idx;
							if (source_idx >= dense_bound)
								dense_tc_XW_one[target_idx] = __float2half(0.0f);
							else
								dense_tc_XW_one[target_idx] = __float2half(XW[source_idx]);
						}

						namedBarrierSync(1, tile_num_per_XW_line * WARP_SIZE);

						wmma::load_matrix_sync(a_frag, sparse_tc_A_one, BLK_W);
						wmma::load_matrix_sync(b_frag, dense_tc_XW_one + wid * BLK_W * BLK_H, BLK_W);
						wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);

						break;
					}
					case 1: {
						unsigned wid_temp = wid - tile_num_per_XW_line;
						#pragma unroll
						for (unsigned idx = laneid; idx < BLK_W * BLK_H; idx += WARP_SIZE){
							dense_rowIdx = sparse_A_tc_ToXW_index_two[idx % BLK_W];
							dense_dimIdx = idx / BLK_W;
							source_idx = dense_rowIdx * embedding_dim + wid_temp * BLK_H + dense_dimIdx;
							target_idx = wid_temp * BLK_W * BLK_H + idx;
							if (source_idx >= dense_bound)
								dense_tc_XW_two[target_idx] = __float2half(0.0f);
							else
								dense_tc_XW_two[target_idx] = __float2half(XW[source_idx]);
						}

						namedBarrierSync(2, tile_num_per_XW_line * WARP_SIZE);

						wmma::load_matrix_sync(a_frag, sparse_tc_A_two, BLK_W);
						wmma::load_matrix_sync(b_frag, dense_tc_XW_two + wid_temp * BLK_W * BLK_H, BLK_W);
						wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);

						break;
					}
					case 2: {
						unsigned wid_temp = wid - 2 * tile_num_per_XW_line;
						#pragma unroll
						for (unsigned idx = laneid; idx < BLK_W * BLK_H; idx += WARP_SIZE){
							dense_rowIdx = sparse_A_tc_ToXW_index_three[idx % BLK_W];
							dense_dimIdx = idx / BLK_W;
							source_idx = dense_rowIdx * embedding_dim + wid_temp * BLK_H + dense_dimIdx;
							target_idx = wid_temp * BLK_W * BLK_H + idx;
							// boundary test.
							if (source_idx >= dense_bound)
								dense_tc_XW_three[target_idx] = __float2half(0.0f);
							else
								dense_tc_XW_three[target_idx] = __float2half(XW[source_idx]);
						}

						namedBarrierSync(3, tile_num_per_XW_line * WARP_SIZE);

						wmma::load_matrix_sync(a_frag, sparse_tc_A_three, BLK_W);
						wmma::load_matrix_sync(b_frag, dense_tc_XW_three + wid_temp * BLK_W * BLK_H, BLK_W);
						wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);

						break;
					}
					case 3: {
						unsigned wid_temp = wid - 3 * tile_num_per_XW_line;
						#pragma unroll
						for (unsigned idx = laneid; idx < BLK_W * BLK_H; idx += WARP_SIZE){
							dense_rowIdx = sparse_A_cd_ToXW_index[idx % BLK_W];
							
							dense_dimIdx = idx / BLK_W;
							
							source_idx = dense_rowIdx * embedding_dim + wid_temp * BLK_H + dense_dimIdx;
							target_idx = wid_temp * BLK_W * BLK_H + idx;
							// boundary test.
							if (source_idx >= dense_bound)
								dense_cd_XW[target_idx] = 0.0f;
							else 
								dense_cd_XW[target_idx] = XW[source_idx];
						}

						namedBarrierSync(4, tile_num_per_XW_line * WARP_SIZE);
						
						if (laneid < BLK_H) {
							unsigned row_id = laneid;
							unsigned col_tmp[BLK_W];
							unsigned non_zero_count = 0;
							#pragma unroll
							for (unsigned a = 0; a < BLK_W; a++) {
								col_tmp[a] = 0;
							}

							unsigned A_index = row_id * (BLK_W + 1);
							#pragma unroll
							for (unsigned a = 0; a < BLK_W; a++) {
								if (sparse_cd_A[A_index + a] == 1.0f) {
									col_tmp[non_zero_count] = a;
									non_zero_count++;
								}
							}

							unsigned output_index = row_id * (BLK_H * tile_num_per_XW_line + 1) + wid_temp * BLK_H;
							float result = 0.0f;
							float output_ori = 0.0f;
							#pragma unroll
							for (unsigned a = 0; a < BLK_H; a++) {
								result = 0.0f;
								for (unsigned b = 0; b < non_zero_count; b++) {
									result += dense_cd_XW[wid_temp * BLK_H * BLK_W + a * BLK_W + col_tmp[b]];
								}
								if (result == 0.0f) break;

								output_ori = output_cd_kernel[output_index + a];
								output_cd_kernel[output_index + a] = output_ori + result;
							}
						}
						break;
					}
					default: break;
				}
			}
			
			else {
				unsigned remain = tile_num_per_block - 4 * i;
				col_nIdx_tc_start = col_nIdx_tc_end_three;
				col_nIdx_tc_end_one = col_nIdx_tc_start + BLK_W;
				col_nIdx_tc_end_two = col_nIdx_tc_end_one + BLK_W;
				col_nIdx_tc_end_three = col_nIdx_tc_end_two + BLK_W;
				
				if (tid < BLK_W){
					sparse_A_cd_ToXW_index[tid] = num_nodes + 1;
					sparse_A_tc_ToXW_index_one[tid] = num_nodes + 1;
					sparse_A_tc_ToXW_index_two[tid] = num_nodes + 1;
					sparse_A_tc_ToXW_index_three[tid] = num_nodes + 1;
				}

				#pragma unroll
				for (unsigned idx = tid; idx < BLK_W * BLK_H; idx += thread_per_block) {
					sparse_tc_A_one[idx] = __float2half(0.0f);
					sparse_tc_A_two[idx] = __float2half(0.0f);
					sparse_tc_A_three[idx] = __float2half(0.0f);
				}

				#pragma unroll
				for (unsigned idx = tid; idx < BLK_W * BLK_H * tile_num_per_XW_line + 8; idx += thread_per_block){
					dense_tc_XW_one[idx] = __float2half(0.0f);
					dense_tc_XW_two[idx] = __float2half(0.0f);
					dense_tc_XW_three[idx] = __float2half(0.0f);
				}

				 __syncthreads();

				#pragma unroll
				for (unsigned eidx = eIdx_start + tid; eidx < eIdx_end; eidx += thread_per_block) {
					unsigned col = edgeToColumn[eidx];
					switch ((col - col_nIdx_tc_start) / BLK_W) {
						case 0: {
							row_local = edgeToRow[eidx] % BLK_H;
							col_local = col % BLK_W;
							sparse_tc_A_one[row_local * BLK_W + col_local] = __float2half(1.0f);
							sparse_A_tc_ToXW_index_one[col_local] = edgeList[eidx];
							break;
						}
						case 1: {
							if (remain <= 1) break;
							row_local = edgeToRow[eidx] % BLK_H;
							col_local = col % BLK_W;
							sparse_tc_A_two[row_local * BLK_W + col_local] = __float2half(1.0f);
							sparse_A_tc_ToXW_index_two[col_local] = edgeList[eidx];
							break;
						}
						case 2: {
							if (remain <= 2) break;
							row_local = edgeToRow[eidx] % BLK_H;
							col_local = col % BLK_W;
							sparse_tc_A_three[row_local * BLK_W + col_local] = __float2half(1.0f);
							sparse_A_tc_ToXW_index_three[col_local] = edgeList[eidx];
							break;
						}
						default: break;
					}
				}

				__syncthreads();

				switch (wid / tile_num_per_XW_line) {
					case 0: {
						#pragma unroll
						for (unsigned idx = laneid; idx < BLK_W * BLK_H; idx += WARP_SIZE){
							dense_rowIdx = sparse_A_tc_ToXW_index_one[idx % BLK_W];
							dense_dimIdx = idx / BLK_W;
							source_idx = dense_rowIdx * embedding_dim + wid * BLK_H + dense_dimIdx;
							target_idx = wid * BLK_W * BLK_H + idx;
							if (source_idx >= dense_bound)
								dense_tc_XW_one[target_idx] = __float2half(0.0f);
							else
								dense_tc_XW_one[target_idx] = __float2half(XW[source_idx]);
						}

						namedBarrierSync(5, tile_num_per_XW_line * WARP_SIZE);

						wmma::load_matrix_sync(a_frag, sparse_tc_A_one, BLK_W);
						wmma::load_matrix_sync(b_frag, dense_tc_XW_one + wid * BLK_W * BLK_H, BLK_W);
						wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
						break;
					}
					case 1: {
						if (remain <= 1) break;
						unsigned wid_temp = wid - tile_num_per_XW_line;
						#pragma unroll
						for (unsigned idx = laneid; idx < BLK_W * BLK_H; idx += WARP_SIZE){
							dense_rowIdx = sparse_A_tc_ToXW_index_two[idx % BLK_W];
							dense_dimIdx = idx / BLK_W;
							source_idx = dense_rowIdx * embedding_dim + wid_temp * BLK_H + dense_dimIdx;
							target_idx = wid_temp * BLK_W * BLK_H + idx;
							if (source_idx >= dense_bound)
								dense_tc_XW_two[target_idx] = __float2half(0.0f);
							else
								dense_tc_XW_two[target_idx] = __float2half(XW[source_idx]);
						}

						namedBarrierSync(6, tile_num_per_XW_line * WARP_SIZE);

						wmma::load_matrix_sync(a_frag, sparse_tc_A_two, BLK_W);
						wmma::load_matrix_sync(b_frag, dense_tc_XW_two + wid_temp * BLK_W * BLK_H, BLK_W);
						wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
						break;
					}
					case 2: {
						if (remain <= 2) break;
						unsigned wid_temp = wid - 2 * tile_num_per_XW_line;
						#pragma unroll
						for (unsigned idx = laneid; idx < BLK_W * BLK_H; idx += WARP_SIZE){
							dense_rowIdx = sparse_A_tc_ToXW_index_three[idx % BLK_W];
							dense_dimIdx = idx / BLK_W;
							source_idx = dense_rowIdx * embedding_dim + wid_temp * BLK_H + dense_dimIdx;
							target_idx = wid_temp * BLK_W * BLK_H + idx;
							
							if (source_idx >= dense_bound)
								dense_tc_XW_three[target_idx] = __float2half(0.0f);
							else
								dense_tc_XW_three[target_idx] = __float2half(XW[source_idx]);
						}

						namedBarrierSync(7, tile_num_per_XW_line * WARP_SIZE);

						wmma::load_matrix_sync(a_frag, sparse_tc_A_three, BLK_W);
						wmma::load_matrix_sync(b_frag, dense_tc_XW_three + wid_temp * BLK_W * BLK_H, BLK_W);
						wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);

						break;
					}
					default: break;
				}
			}
		
			__syncthreads();
		}

		#pragma unroll
		for (unsigned idx = tid; idx < BLK_W * BLK_H * tile_num_per_XW_line + 4; idx += thread_per_block){
			output_tc_kernel_three[idx] = 0.0f;
		}
		__syncthreads();

		switch (wid / tile_num_per_XW_line) {
			case 0: {
				wmma::store_matrix_sync(output_tc_kernel_one + wid * BLK_H, acc_frag, tile_num_per_XW_line * BLK_H, wmma::mem_row_major);
				break;
			}
			case 1: {
				wmma::store_matrix_sync(output_tc_kernel_two + (wid - tile_num_per_XW_line) * BLK_H, acc_frag, tile_num_per_XW_line * BLK_H, wmma::mem_row_major);
				break;
			}
			case 2: {
				wmma::store_matrix_sync(output_tc_kernel_three + (wid - 2 * tile_num_per_XW_line) * BLK_H, acc_frag, tile_num_per_XW_line * BLK_H, wmma::mem_row_major);
				break;
			}
			default: break;
		}

		__syncthreads();

		unsigned output_bid_first_index = bid * BLK_H * embedding_dim;
		unsigned output_tc_kernel_index = 0;
		unsigned output_cd_kernel_index = 0;
		#pragma unroll
		for (unsigned idx = tid; idx < BLK_H * embedding_dim; idx += thread_per_block) {
			output_tc_kernel_index = idx + (idx / embedding_dim) * (tile_num_per_XW_line * BLK_H - embedding_dim);
			output_cd_kernel_index =  idx + (idx / embedding_dim) * (tile_num_per_XW_line * BLK_H + 1 - embedding_dim);
			output[output_bid_first_index + idx] = output_tc_kernel_one[output_tc_kernel_index] +
												   output_tc_kernel_two[output_tc_kernel_index] +
												   output_tc_kernel_three[output_tc_kernel_index] +
												   output_cd_kernel[output_cd_kernel_index];
		}
		
		__syncthreads();
	}
}

__device__ __forceinline__ void namedBarrierSync(int name, int numThreads) {
      asm volatile("bar.sync %0, %1;" : : "r"(name), "r"(numThreads) : "memory");
}



