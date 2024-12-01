#include <stdio.h>
#include <stdlib.h>
#include "Sorter.h"
#include <bitset>
#include <iostream>

#define BITS_LEN 32
#define BLOCK_SIZE 1024
#define MAX_LAYER 32
#define LAYER_SIZE 2 * BLOCK_SIZE // always multiple of BLOCK SIZE
// Bank Conflict
#define NUM_BANKS 32
#define LOG_NUM_BANKS 5
#define CONFLICT_FREE_OFFSET(n)((n) >> LOG_NUM_BANKS)

namespace v4 {

__global__ void preprocess_float(DEBUG_FLOAT* data) {
	int idx  = blockIdx.x * blockDim.x + threadIdx.x;
 	unsigned int data_temp = *(unsigned int *)(&data[idx]);    
    data_temp = (data_temp >> 31 & 0x1)? ~(data_temp): (data_temp) | 0x80000000;
	data[idx] = *(DEBUG_FLOAT *)&data_temp; 
}

__global__ void postprocess_float(DEBUG_FLOAT* const data) {
	int idx  = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int data_temp = *(unsigned int *)(&data[idx]);
    data_temp = (data_temp >> 31 & 0x1)? (data_temp) & 0x7fffffff: ~(data_temp);
	data[idx] = *(DEBUG_FLOAT *)&data_temp;
}

__global__ void reorder(
	DEBUG_FLOAT* data , 
	DEBUG_FLOAT* data_origin , 
	int n , 
	unsigned int* prefixSum_1_buffer,
	unsigned int* prefixSum_0_buffer,
	unsigned int* histgram_buffer,
	int num_layer,
	unsigned int *layer_offset,
	int sort_bit
	)
{
	int tid = threadIdx.x;
	int bid = blockIdx.x * blockDim.x + threadIdx.x;

	__shared__ DEBUG_FLOAT sdata[BLOCK_SIZE];
	__shared__ unsigned int slayer_offset[MAX_LAYER];
	sdata[tid] = data_origin[bid];
	if (tid < num_layer) slayer_offset[tid] = layer_offset[tid];
	__syncthreads();

	bool is_one = (__float_as_uint(sdata[tid]) >> sort_bit) & 1;
	int reordered_idx = histgram_buffer[1] * is_one;

	for (int layer = 0; layer < num_layer; ++layer) {
		if (is_one) reordered_idx += prefixSum_1_buffer[bid + slayer_offset[layer]];
		else reordered_idx += prefixSum_0_buffer[bid + slayer_offset[layer]];
		bid /= LAYER_SIZE;
	}					
	data[reordered_idx] = sdata[tid];
}


__global__ void init_sort(
	DEBUG_FLOAT* data_origin,
	unsigned int* prefixSum_1_buffer, 
	unsigned int* prefixSum_0_buffer,
	unsigned int sort_bit) 
{
	int tid = threadIdx.x;
	int bid  = blockIdx.x * blockDim.x + threadIdx.x;

	__shared__ DEBUG_FLOAT sdata[BLOCK_SIZE];
	sdata[tid] = data_origin[bid];

	bool is_one = (__float_as_uint(sdata[tid]) >> sort_bit) & 1;
	prefixSum_1_buffer[bid] = is_one;
	prefixSum_0_buffer[bid] = !is_one;
}


__global__ void block_prefix_sum(
	unsigned int* prefixSum_1_buffer, 
	unsigned int* prefixSum_0_buffer,
	unsigned int in_offset,
	unsigned int out_offset,
	unsigned int* histgram_buffer,
	bool isLast,
	unsigned int bit) 
{
	int tid = threadIdx.x;
	int bid = blockIdx.x * LAYER_SIZE;
	unsigned int last_1 = 0;
	unsigned int last_0 = 0;
	
	__shared__ unsigned int s1[LAYER_SIZE + CONFLICT_FREE_OFFSET(LAYER_SIZE)];
	__shared__ unsigned int s0[LAYER_SIZE + CONFLICT_FREE_OFFSET(LAYER_SIZE)];

	// Load Data
	int ai = tid, bi = tid + (LAYER_SIZE / 2);
	int bankOffsetA = CONFLICT_FREE_OFFSET(ai), bankOffsetB = CONFLICT_FREE_OFFSET(bi);

	s1[ai + bankOffsetA] = prefixSum_1_buffer[ai + bid + in_offset];
	s1[bi + bankOffsetB] = prefixSum_1_buffer[bi + bid + in_offset];

	s0[ai + bankOffsetA] = prefixSum_0_buffer[ai + bid + in_offset];
	s0[bi + bankOffsetB] = prefixSum_0_buffer[bi + bid + in_offset];

	if (tid == BLOCK_SIZE - 1) {
		last_1 = prefixSum_1_buffer[bi + bid + in_offset];
		last_0 = prefixSum_0_buffer[bi + bid + in_offset];
	}
	
	// Up-Sweep
	int offset = 1;
	for (int d = LAYER_SIZE >> 1; d > 0; d >>= 1) {
		__syncthreads();
		if (tid < d) {
			int ai = offset * (2 * tid + 1) - 1;
			int bi = offset * (2 * tid + 2) - 1;
			ai += CONFLICT_FREE_OFFSET(ai);
			bi += CONFLICT_FREE_OFFSET(bi);
			s1[bi] += s1[ai];
			s0[bi] += s0[ai];
		}
		offset <<= 1;
	}

	// Down-Sweep
	if (tid == 0) {
		s1[LAYER_SIZE - 1 + CONFLICT_FREE_OFFSET(LAYER_SIZE - 1)] = 0;
		s0[LAYER_SIZE - 1 + CONFLICT_FREE_OFFSET(LAYER_SIZE - 1)] = 0;
	}
	
	for (int d = 1; d < LAYER_SIZE; d <<= 1) {
		offset >>= 1;
		__syncthreads();
		if (tid < d) {
			int ai = offset * (2 * tid + 1) - 1;
			int bi = offset * (2 * tid + 2) - 1;
			ai += CONFLICT_FREE_OFFSET(ai);
			bi += CONFLICT_FREE_OFFSET(bi);

			unsigned int temp;
			temp = s1[ai];
			s1[ai] = s1[bi];
			s1[bi] += temp;
			
			temp = s0[ai];
			s0[ai] = s0[bi];
			s0[bi] += temp;
		}
	}
	__syncthreads();
	

	if (tid == BLOCK_SIZE - 1) {
		if (!isLast) {
			prefixSum_1_buffer[out_offset + blockIdx.x] = s1[LAYER_SIZE - 1 + CONFLICT_FREE_OFFSET(LAYER_SIZE - 1)] + last_1;
			prefixSum_0_buffer[out_offset + blockIdx.x] = s0[LAYER_SIZE - 1 + CONFLICT_FREE_OFFSET(LAYER_SIZE - 1)] + last_0;
		} else {
			histgram_buffer[1] = s0[LAYER_SIZE - 1 + CONFLICT_FREE_OFFSET(LAYER_SIZE - 1)] + last_0;
		}
	}

	prefixSum_1_buffer[ai + bid + in_offset] = s1[ai + bankOffsetA];
	prefixSum_1_buffer[bi + bid + in_offset] = s1[bi + bankOffsetB];
	prefixSum_0_buffer[ai + bid + in_offset] = s0[ai + bankOffsetA];
	prefixSum_0_buffer[bi + bid + in_offset] = s0[bi + bankOffsetB];
}

}

class RadixSorterv4 :public Sorter
{
	// Layerwise Prefix sum with Work-Efficient Sum Scan
	// Avoid Bank Conflict
public:
	
	void sort(DEBUG_FLOAT*& datas , int data_length) override{

		// Padding 
		int n = ceil(data_length, LAYER_SIZE) * LAYER_SIZE;
		int pad_len = n - data_length;

		// Init memory
		DEBUG_FLOAT *dev_data[2] = {0};		
		unsigned int *dev_histgram = 0;
		unsigned int *dev_layer_offset = 0;
		unsigned int *dev_1prefixSum = 0;
		unsigned int *dev_0prefixSum = 0;
		int pout = 0, pin = 1;

		// Buffer	    
		cudaMalloc((void**)&dev_data[pout], n * sizeof(DEBUG_FLOAT));
		cudaMemset(dev_data[pout] + data_length, 0xff,  pad_len * sizeof(DEBUG_FLOAT));
		cudaMemcpy(dev_data[pout], datas, data_length * sizeof(DEBUG_FLOAT), cudaMemcpyHostToDevice);

	    cudaMalloc((void**)&dev_data[pin], n * sizeof(DEBUG_FLOAT));
		cudaMemset(dev_data[pin] + data_length, 0xff,  pad_len * sizeof(DEBUG_FLOAT));
		cudaMemcpy(dev_data[pin], datas, data_length * sizeof(DEBUG_FLOAT), cudaMemcpyHostToDevice);

		// Histgram and prefix sum
		unsigned int prefix_sum_size = 0;
		unsigned int layer_offset[MAX_LAYER];
		int num_layer = 0;

		// Layer Initialization
		// n -> log_1024 (n) -> ... -> 1
		layer_offset[0] = 0;
		for (int layer_size = ceil(n, LAYER_SIZE); ; layer_size = ceil(layer_size, LAYER_SIZE)) {
			prefix_sum_size += layer_size * LAYER_SIZE;
			++num_layer;
			layer_offset[num_layer] = layer_offset[num_layer - 1] + layer_size * LAYER_SIZE;
			if (layer_size == 1) break;
		}
		prefix_sum_size++;


		cudaMalloc((void**)&dev_histgram, 2 * sizeof(int));
		cudaMalloc((void**)&dev_layer_offset, (num_layer + 1) * sizeof(int));
		cudaMalloc((void**)&dev_1prefixSum, prefix_sum_size * sizeof(int));
		cudaMalloc((void**)&dev_0prefixSum, prefix_sum_size * sizeof(int));
		
		// Initialize the buffer
		cudaMemcpy(dev_layer_offset, layer_offset, (num_layer + 1) * sizeof(int), cudaMemcpyHostToDevice);


		// Kernel launch
		int numBlocks = ceil(n, BLOCK_SIZE);

		// Prepare index and sortable conversion
		v4::preprocess_float<<< numBlocks , BLOCK_SIZE >>>(dev_data[pout]);
		
		for(int i = 0; i < BITS_LEN; ++i) {
			pout = 1 - pout; // swap double buffer indices
			pin = 1 - pout;

			cudaMemset(dev_histgram , 0, 2 * sizeof(int));
			cudaMemset(dev_1prefixSum, 0, prefix_sum_size * sizeof(int));
			cudaMemset(dev_0prefixSum, 0, prefix_sum_size * sizeof(int));
			cudaDeviceSynchronize();

			v4::init_sort<<< numBlocks , BLOCK_SIZE >>>(dev_data[pin],
														dev_1prefixSum,
														dev_0prefixSum,
														i);
			
			// Prefix_sum
			int num_prefix_sum_block = n;		
			
			for (int layer = 0; layer < num_layer; ++layer) {
				num_prefix_sum_block = ceil(num_prefix_sum_block, LAYER_SIZE);
				v4::block_prefix_sum<<< num_prefix_sum_block, BLOCK_SIZE >>>(dev_1prefixSum, 
																		dev_0prefixSum, 
																		layer_offset[layer], 
																		layer_offset[layer + 1], 
																		dev_histgram,
																		layer == num_layer - 1,
																		i);
			}
			// Sort
			v4::reorder<<< numBlocks , BLOCK_SIZE >>>(dev_data[pout],
													  dev_data[pin], 
													  n, 
													  dev_1prefixSum, 
													  dev_0prefixSum, 
													  dev_histgram,
													  num_layer,
													  dev_layer_offset, 
													  i);
			
		}
		
		
		// Copy origin_datas into sortable by index.
		v4::postprocess_float<<< numBlocks , BLOCK_SIZE >>>(dev_data[pout]);

		// Clear kernel
		cudaMemcpy(datas, dev_data[pout] + pad_len, data_length * sizeof(float), cudaMemcpyDeviceToHost);
		
		cudaFree(dev_data[pout]);
		cudaFree(dev_data[pin]);
		cudaFree(dev_0prefixSum);
		cudaFree(dev_1prefixSum);
		cudaFree(dev_histgram);
		cudaFree(dev_layer_offset);
		/*
		printf("============================= datas =================================\n");
		for(int i = 0 ; i < data_length ; i++){
			printBits(datas[i] );			
			printf(" : %f" , datas[i] );
			
    		printf("\n");
		}
		printf("\n");
		*/
	}
	
private:
	void printBits(DEBUG_FLOAT num) {
		  unsigned int bits = *reinterpret_cast<unsigned int*>(&num);

		// Print the bits
		std::bitset<32> bitset(bits);
		std::cout << "Bits of " << num << ": " << bitset << std::endl;
	}

	inline int ceil(int y, int b) { return (y + b - 1) / b; }
};
