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
#define INDEX(n)((n) + ((n) >> LOG_NUM_BANKS))

namespace v5 {

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
	int bid  = blockIdx.x * blockDim.x + threadIdx.x;

	bool is_one = (__float_as_uint(data_origin[bid]) >> sort_bit) & 1;
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
	
	__shared__ unsigned int s1[INDEX(LAYER_SIZE)];
	__shared__ unsigned int s0[INDEX(LAYER_SIZE)];

	// Load Data
	int ai = tid, bi = tid + (LAYER_SIZE / 2);

	s1[INDEX(ai)] = prefixSum_1_buffer[ai + bid + in_offset];
	s1[INDEX(bi)] = prefixSum_1_buffer[bi + bid + in_offset];

	s0[INDEX(ai)] = prefixSum_0_buffer[ai + bid + in_offset];
	s0[INDEX(bi)] = prefixSum_0_buffer[bi + bid + in_offset];

	if (tid == BLOCK_SIZE - 1) {
		last_1 = prefixSum_1_buffer[bi + bid + in_offset];
		last_0 = prefixSum_0_buffer[bi + bid + in_offset];
	}
	__syncthreads();

	// Up-Sweep
	if (tid < 1024) { // offset = 1
		s1[INDEX((tid << 1) + 1)] += s1[INDEX((tid << 1))];
		s0[INDEX((tid << 1) + 1)] += s0[INDEX((tid << 1))];
	}
	__syncthreads();
	if (tid < 512) { // offset = 2
		s1[INDEX((tid << 2) + 3)] += s1[INDEX((tid << 2) + 1)];
		s0[INDEX((tid << 2) + 3)] += s0[INDEX((tid << 2) + 1)];
	}
	__syncthreads();
	if (tid < 256) { // offset = 4
		s1[INDEX((tid << 3) + 7)] += s1[INDEX((tid << 3) + 3)];
		s0[INDEX((tid << 3) + 7)] += s0[INDEX((tid << 3) + 3)];
	}
	__syncthreads();
	if (tid < 128) { 
		s1[INDEX((tid << 4) + 15)] += s1[INDEX((tid << 4) + 7)];
		s0[INDEX((tid << 4) + 15)] += s0[INDEX((tid << 4) + 7)];
	}
	__syncthreads();
	if (tid < 64) {
		s1[INDEX((tid << 5) + 31)] += s1[INDEX((tid << 5) + 15)];
		s0[INDEX((tid << 5) + 31)] += s0[INDEX((tid << 5) + 15)];
	}
	__syncthreads();
	// tid < 32
	if (tid < 32) {
		s1[INDEX((tid << 6) + 63)] += s1[INDEX((tid << 6) + 31)];
		s0[INDEX((tid << 6) + 63)] += s0[INDEX((tid << 6) + 31)];
	}
	// tid < 16
	if (tid < 16) {
		s1[INDEX((tid << 7) + 127)] += s1[INDEX((tid << 7) + 63)];
		s0[INDEX((tid << 7) + 127)] += s0[INDEX((tid << 7) + 63)];
	}
	// tid < 8
	if (tid < 8) {
		s1[INDEX((tid << 8) + 255)] += s1[INDEX((tid << 8) + 127)];
		s0[INDEX((tid << 8) + 255)] += s0[INDEX((tid << 8) + 127)];
	}
	// tid < 4
	if (tid < 4) {
		s1[INDEX((tid << 9) + 511)] += s1[INDEX((tid << 9) + 255)];
		s0[INDEX((tid << 9) + 511)] += s0[INDEX((tid << 9) + 255)];
	}
	// tid < 2
	if (tid < 2) {
		s1[INDEX((tid << 10) + 1023)] += s1[INDEX((tid << 10) + 511)];
		s0[INDEX((tid << 10) + 1023)] += s0[INDEX((tid << 10) + 511)];
	}
	// tid < 1
	if (tid < 1) {
		s1[INDEX(LAYER_SIZE - 1)] = 0;
		s0[INDEX(LAYER_SIZE - 1)] = 0;
	}

	// Down-Sweep
	unsigned int temp;
	if (tid < 1) {
		temp = s1[INDEX((tid << 11) + 1023)];
		s1[INDEX((tid << 11) + 1023)] = s1[INDEX((tid << 11) + 2047)];
		s1[INDEX((tid << 11) + 2047)] += temp;
		temp = s0[INDEX((tid << 11) + 1023)];
		s0[INDEX((tid << 11) + 1023)] = s0[INDEX((tid << 11) + 2047)];
		s0[INDEX((tid << 11) + 2047)] += temp;
	}
	if (tid < 2) {
		temp = s1[INDEX((tid << 10) + 511)];
		s1[INDEX((tid << 10) + 511)] = s1[INDEX((tid << 10) + 1023)];
		s1[INDEX((tid << 10) + 1023)] += temp;
		temp = s0[INDEX((tid << 10) + 511)];
		s0[INDEX((tid << 10) + 511)] = s0[INDEX((tid << 10) + 1023)];
		s0[INDEX((tid << 10) + 1023)] += temp;
	}
	if (tid < 4) {
		temp = s1[INDEX((tid << 9) + 255)];
		s1[INDEX((tid << 9) + 255)] = s1[INDEX((tid << 9) + 511)];
		s1[INDEX((tid << 9) + 511)] += temp;
		temp = s0[INDEX((tid << 9) + 255)];
		s0[INDEX((tid << 9) + 255)] = s0[INDEX((tid << 9) + 511)];
		s0[INDEX((tid << 9) + 511)] += temp;
	}
	if (tid < 8) {
		temp = s1[INDEX((tid << 8) + 127)];
		s1[INDEX((tid << 8) + 127)] = s1[INDEX((tid << 8) + 255)];
		s1[INDEX((tid << 8) + 255)] += temp;
		temp = s0[INDEX((tid << 8) + 127)];
		s0[INDEX((tid << 8) + 127)] = s0[INDEX((tid << 8) + 255)];
		s0[INDEX((tid << 8) + 255)] += temp;
	}
	if (tid < 16) {
		temp = s1[INDEX((tid << 7) + 63)];
		s1[INDEX((tid << 7) + 63)] = s1[INDEX((tid << 7) + 127)];
		s1[INDEX((tid << 7) + 127)] += temp;
		temp = s0[INDEX((tid << 7) + 63)];
		s0[INDEX((tid << 7) + 63)] = s0[INDEX((tid << 7) + 127)];
		s0[INDEX((tid << 7) + 127)] += temp;
	}
	if (tid < 32) {
		temp = s1[INDEX((tid << 6) + 31)];
		s1[INDEX((tid << 6) + 31)] = s1[INDEX((tid << 6) + 63)];
		s1[INDEX((tid << 6) + 63)] += temp;
		temp = s0[INDEX((tid << 6) + 31)];
		s0[INDEX((tid << 6) + 31)] = s0[INDEX((tid << 6) + 63)];
		s0[INDEX((tid << 6) + 63)] += temp;
	}
	__syncthreads();
	if (tid < 64) {
		temp = s1[INDEX((tid << 5) + 15)];
		s1[INDEX((tid << 5) + 15)] = s1[INDEX((tid << 5) + 31)];
		s1[INDEX((tid << 5) + 31)] += temp;
		temp = s0[INDEX((tid << 5) + 15)];
		s0[INDEX((tid << 5) + 15)] = s0[INDEX((tid << 5) + 31)];
		s0[INDEX((tid << 5) + 31)] += temp;
	}
	__syncthreads();
	if (tid < 128) {
		temp = s1[INDEX((tid << 4) + 7)];
		s1[INDEX((tid << 4) + 7)] = s1[INDEX((tid << 4) + 15)];
		s1[INDEX((tid << 4) + 15)] += temp;
		temp = s0[INDEX((tid << 4) + 7)];
		s0[INDEX((tid << 4) + 7)] = s0[INDEX((tid << 4) + 15)];
		s0[INDEX((tid << 4) + 15)] += temp;
	}
	__syncthreads();
	if (tid < 256) {
		temp = s1[INDEX((tid << 3) + 3)];
		s1[INDEX((tid << 3) + 3)] = s1[INDEX((tid << 3) + 7)];
		s1[INDEX((tid << 3) + 7)] += temp;
		temp = s0[INDEX((tid << 3) + 3)];
		s0[INDEX((tid << 3) + 3)] = s0[INDEX((tid << 3) + 7)];
		s0[INDEX((tid << 3) + 7)] += temp;
	}
	__syncthreads();
	if (tid < 512) {
		temp = s1[INDEX((tid << 2) + 1)];
		s1[INDEX((tid << 2) + 1)] = s1[INDEX((tid << 2) + 3)];
		s1[INDEX((tid << 2) + 3)] += temp;
		temp = s0[INDEX((tid << 2) + 1)];
		s0[INDEX((tid << 2) + 1)] = s0[INDEX((tid << 2) + 3)];
		s0[INDEX((tid << 2) + 3)] += temp;
	}
	__syncthreads();
	if (tid < 1024) {
		temp = s1[INDEX((tid << 1))];
		s1[INDEX((tid << 1))] = s1[INDEX((tid << 1) + 1)];
		s1[INDEX((tid << 1) + 1)] += temp;
		temp = s0[INDEX((tid << 1))];
		s0[INDEX((tid << 1))] = s0[INDEX((tid << 1) + 1)];
		s0[INDEX((tid << 1) + 1)] += temp;
	}
	__syncthreads();
	

	if (tid == BLOCK_SIZE - 1) {
		if (!isLast) {
			prefixSum_1_buffer[out_offset + blockIdx.x] = s1[INDEX(LAYER_SIZE - 1)] + last_1;
			prefixSum_0_buffer[out_offset + blockIdx.x] = s0[INDEX(LAYER_SIZE - 1)] + last_0;
		} else {
			histgram_buffer[1] = s0[INDEX(LAYER_SIZE - 1)] + last_0;
		}
	}

	prefixSum_1_buffer[ai + bid + in_offset] = s1[INDEX(ai)];
	prefixSum_1_buffer[bi + bid + in_offset] = s1[INDEX(bi)];
	prefixSum_0_buffer[ai + bid + in_offset] = s0[INDEX(ai)];
	prefixSum_0_buffer[bi + bid + in_offset] = s0[INDEX(bi)];
}

}

class RadixSorterv5 :public Sorter
{
	// Layerwise Prefix sum with Work-Efficient Sum Scan
	// Avoid Bank Conflict
	// Unroll
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
		v5::preprocess_float<<< numBlocks , BLOCK_SIZE >>>(dev_data[pout]);
		
		for(int i = 0; i < BITS_LEN; ++i) {
			pout = 1 - pout; // swap double buffer indices
			pin = 1 - pout;

			cudaMemset(dev_histgram , 0, 2 * sizeof(int));
			cudaMemset(dev_1prefixSum, 0, prefix_sum_size * sizeof(int));
			cudaMemset(dev_0prefixSum, 0, prefix_sum_size * sizeof(int));
			cudaDeviceSynchronize();

			v5::init_sort<<< numBlocks , BLOCK_SIZE >>>(dev_data[pin],
														dev_1prefixSum,
														dev_0prefixSum,
														i);
			
			// Prefix_sum
			int num_prefix_sum_block = n;		
			
			for (int layer = 0; layer < num_layer; ++layer) {
				num_prefix_sum_block = ceil(num_prefix_sum_block, LAYER_SIZE);
				v5::block_prefix_sum<<< num_prefix_sum_block, BLOCK_SIZE >>>(dev_1prefixSum, 
																		dev_0prefixSum, 
																		layer_offset[layer], 
																		layer_offset[layer + 1], 
																		dev_histgram,
																		layer == num_layer - 1,
																		i);
			}
			// Sort
			v5::reorder<<< numBlocks , BLOCK_SIZE >>>(dev_data[pout],
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
		v5::postprocess_float<<< numBlocks , BLOCK_SIZE >>>(dev_data[pout]);

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
