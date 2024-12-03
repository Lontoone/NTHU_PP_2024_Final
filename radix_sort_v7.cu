#include <stdio.h>
#include <stdlib.h>
#include "Sorter.h"
#include <bitset>
#include <iostream>

#define BIT_RANGE 4 // 2 ^ BIT_NUM
#define BIT_NUM 2
#define BITS_LEN 16 // BITS_LEN * BIT_NUM = 32
#define BLOCK_SIZE 1024
#define MAX_LAYER 32
#define LAYER_SIZE 2048 // always multiple of BLOCK SIZE
// Bank Conflict
#define NUM_BANKS 32
#define LOG_NUM_BANKS 5
#define INDEX(n)((n) + ((n) >> LOG_NUM_BANKS))

namespace v7 {

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
	unsigned int** prefixSum_buffer,
	unsigned int* histgram_buffer,
	int num_layer,
	unsigned int *layer_offset,
	int sort_bit
	)
{
	int tid = threadIdx.x;
	int bid = blockIdx.x * LAYER_SIZE + threadIdx.x;

	__shared__ DEBUG_FLOAT sdata[LAYER_SIZE];
	__shared__ unsigned int slayer_offset[MAX_LAYER];
	__shared__ unsigned int slayer_prefixSum[BIT_RANGE][MAX_LAYER];
	sdata[tid] = data_origin[bid];
	sdata[tid + LAYER_SIZE / 2] = data_origin[bid + LAYER_SIZE / 2];

	if (tid < num_layer) slayer_offset[tid] = layer_offset[tid];
	__syncthreads();

	if (tid < 4 * (num_layer - 1)) {
		int idx = bid, group = tid >> 2, layer = 0;
		for (layer = 0; layer < (num_layer - 1) && layer <= group; ++layer) {
			idx /= LAYER_SIZE;
		}			
		slayer_prefixSum[tid & 0b11][group] = prefixSum_buffer[tid & 0b11][idx + slayer_offset[layer]];
	}
	__syncthreads();

	int val0 = (__float_as_uint(sdata[tid]) >> sort_bit * BIT_NUM) & 0b11;
	int val1 = (__float_as_uint(sdata[tid + LAYER_SIZE / 2]) >> sort_bit * BIT_NUM) & 0b11;
	int reordered_idx0 = histgram_buffer[val0] + prefixSum_buffer[val0][bid];
	int reordered_idx1 = histgram_buffer[val1] + prefixSum_buffer[val1][bid + LAYER_SIZE / 2];

	for (int layer = 0; layer < num_layer - 1; ++layer) {
		reordered_idx0 += slayer_prefixSum[val0][layer];
		reordered_idx1 += slayer_prefixSum[val1][layer];
	}					
	data[reordered_idx0] = sdata[tid];
	data[reordered_idx1] = sdata[tid + LAYER_SIZE / 2];
}


__global__ void init_sort(
	DEBUG_FLOAT* data_origin,
	unsigned int** prefixSum_buffer, 
	unsigned int sort_bit) 
{
	int bid  = blockIdx.x * blockDim.x + threadIdx.x;

	int val = (__float_as_uint(data_origin[bid]) >> sort_bit * BIT_NUM) & 0b11;
	prefixSum_buffer[0][bid] = (val == 0);
	prefixSum_buffer[1][bid] = (val == 1);
	prefixSum_buffer[2][bid] = (val == 2);
	prefixSum_buffer[3][bid] = (val == 3);
}

__global__ void block_prefix_sum(
	unsigned int** prefixSum_buffer, 
	unsigned int in_offset,
	unsigned int out_offset,
	unsigned int* histgram_buffer,
	bool isLast,
	unsigned int bit) 
{
	int tid = threadIdx.x;
	int bid = blockIdx.x * LAYER_SIZE;
	unsigned int last_0 = 0, last_1 = 0, last_2 = 0, last_3 = 0;
	
	__shared__ unsigned int s[BIT_RANGE][INDEX(LAYER_SIZE)];

	// Load Data
	int ai = tid, bi = tid + (LAYER_SIZE / 2);

	s[0][INDEX(ai)] = prefixSum_buffer[0][ai + bid + in_offset];
	s[0][INDEX(bi)] = prefixSum_buffer[0][bi + bid + in_offset];

	s[1][INDEX(ai)] = prefixSum_buffer[1][ai + bid + in_offset];
	s[1][INDEX(bi)] = prefixSum_buffer[1][bi + bid + in_offset];

	s[2][INDEX(ai)] = prefixSum_buffer[2][ai + bid + in_offset];
	s[2][INDEX(bi)] = prefixSum_buffer[2][bi + bid + in_offset];
	
	s[3][INDEX(ai)] = prefixSum_buffer[3][ai + bid + in_offset];
	s[3][INDEX(bi)] = prefixSum_buffer[3][bi + bid + in_offset];

	if (tid == BLOCK_SIZE - 1) {
		last_0 = prefixSum_buffer[0][bi + bid + in_offset];
		last_1 = prefixSum_buffer[1][bi + bid + in_offset];
		last_2 = prefixSum_buffer[2][bi + bid + in_offset];
		last_3 = prefixSum_buffer[3][bi + bid + in_offset];
	}
	__syncthreads();

	// Up-Sweep
	if (tid < 1024) { // offset = 1
		s[0][INDEX((tid << 1) + 1)] += s[0][INDEX((tid << 1))];
		s[1][INDEX((tid << 1) + 1)] += s[1][INDEX((tid << 1))];
		s[2][INDEX((tid << 1) + 1)] += s[2][INDEX((tid << 1))];
		s[3][INDEX((tid << 1) + 1)] += s[3][INDEX((tid << 1))];
	}
	__syncthreads();
	if (tid < 512) { // offset = 2
		s[0][INDEX((tid << 2) + 3)] += s[0][INDEX((tid << 2) + 1)];
		s[1][INDEX((tid << 2) + 3)] += s[1][INDEX((tid << 2) + 1)];
		s[2][INDEX((tid << 2) + 3)] += s[2][INDEX((tid << 2) + 1)];
		s[3][INDEX((tid << 2) + 3)] += s[3][INDEX((tid << 2) + 1)];
	}
	__syncthreads();
	if (tid < 256) { // offset = 4
		s[0][INDEX((tid << 3) + 7)] += s[0][INDEX((tid << 3) + 3)];
		s[1][INDEX((tid << 3) + 7)] += s[1][INDEX((tid << 3) + 3)];
		s[2][INDEX((tid << 3) + 7)] += s[2][INDEX((tid << 3) + 3)];
		s[3][INDEX((tid << 3) + 7)] += s[3][INDEX((tid << 3) + 3)];
	}
	__syncthreads();
	if (tid < 128) { 
		s[0][INDEX((tid << 4) + 15)] += s[0][INDEX((tid << 4) + 7)];
		s[1][INDEX((tid << 4) + 15)] += s[1][INDEX((tid << 4) + 7)];
		s[2][INDEX((tid << 4) + 15)] += s[2][INDEX((tid << 4) + 7)];
		s[3][INDEX((tid << 4) + 15)] += s[3][INDEX((tid << 4) + 7)];
	}
	__syncthreads();
	if (tid < 64) {
		s[0][INDEX((tid << 5) + 31)] += s[0][INDEX((tid << 5) + 15)];
		s[1][INDEX((tid << 5) + 31)] += s[1][INDEX((tid << 5) + 15)];
		s[2][INDEX((tid << 5) + 31)] += s[2][INDEX((tid << 5) + 15)];
		s[3][INDEX((tid << 5) + 31)] += s[3][INDEX((tid << 5) + 15)];
	}
	__syncthreads();
	// tid < 32
	if (tid < 32) {
		s[0][INDEX((tid << 6) + 63)] += s[0][INDEX((tid << 6) + 31)];
		s[1][INDEX((tid << 6) + 63)] += s[1][INDEX((tid << 6) + 31)];
		s[2][INDEX((tid << 6) + 63)] += s[2][INDEX((tid << 6) + 31)];
		s[3][INDEX((tid << 6) + 63)] += s[3][INDEX((tid << 6) + 31)];
	}
	// tid < 16
	if (tid < 16) {
		s[0][INDEX((tid << 7) + 127)] += s[0][INDEX((tid << 7) + 63)];
		s[1][INDEX((tid << 7) + 127)] += s[1][INDEX((tid << 7) + 63)];
		s[2][INDEX((tid << 7) + 127)] += s[2][INDEX((tid << 7) + 63)];
		s[3][INDEX((tid << 7) + 127)] += s[3][INDEX((tid << 7) + 63)];
	}
	// tid < 8
	if (tid < 8) {
		s[0][INDEX((tid << 8) + 255)] += s[0][INDEX((tid << 8) + 127)];
		s[1][INDEX((tid << 8) + 255)] += s[1][INDEX((tid << 8) + 127)];
		s[2][INDEX((tid << 8) + 255)] += s[2][INDEX((tid << 8) + 127)];
		s[3][INDEX((tid << 8) + 255)] += s[3][INDEX((tid << 8) + 127)];
	}
	// tid < 4
	if (tid < 4) {
		s[0][INDEX((tid << 9) + 511)] += s[0][INDEX((tid << 9) + 255)];
		s[1][INDEX((tid << 9) + 511)] += s[1][INDEX((tid << 9) + 255)];
		s[2][INDEX((tid << 9) + 511)] += s[2][INDEX((tid << 9) + 255)];
		s[3][INDEX((tid << 9) + 511)] += s[3][INDEX((tid << 9) + 255)];
	}
	// tid < 2
	if (tid < 2) {
		s[0][INDEX((tid << 10) + 1023)] += s[0][INDEX((tid << 10) + 511)];
		s[1][INDEX((tid << 10) + 1023)] += s[1][INDEX((tid << 10) + 511)];
		s[2][INDEX((tid << 10) + 1023)] += s[2][INDEX((tid << 10) + 511)];
		s[3][INDEX((tid << 10) + 1023)] += s[3][INDEX((tid << 10) + 511)];
	}
	// tid < 1
	if (tid < 1) {
		s[0][INDEX(LAYER_SIZE - 1)] = 0;
		s[1][INDEX(LAYER_SIZE - 1)] = 0;
		s[2][INDEX(LAYER_SIZE - 1)] = 0;
		s[3][INDEX(LAYER_SIZE - 1)] = 0;
	}

	// Down-Sweep
	unsigned int temp;
	if (tid < 1) {
		temp = s[0][INDEX((tid << 11) + 1023)];
		s[0][INDEX((tid << 11) + 1023)] = s[0][INDEX((tid << 11) + 2047)];
		s[0][INDEX((tid << 11) + 2047)] += temp;
		temp = s[1][INDEX((tid << 11) + 1023)];
		s[1][INDEX((tid << 11) + 1023)] = s[1][INDEX((tid << 11) + 2047)];
		s[1][INDEX((tid << 11) + 2047)] += temp;
		temp = s[2][INDEX((tid << 11) + 1023)];
		s[2][INDEX((tid << 11) + 1023)] = s[2][INDEX((tid << 11) + 2047)];
		s[2][INDEX((tid << 11) + 2047)] += temp;
		temp = s[3][INDEX((tid << 11) + 1023)];
		s[3][INDEX((tid << 11) + 1023)] = s[3][INDEX((tid << 11) + 2047)];
		s[3][INDEX((tid << 11) + 2047)] += temp;
	}
	if (tid < 2) {
		temp = s[0][INDEX((tid << 10) + 511)];
		s[0][INDEX((tid << 10) + 511)] = s[0][INDEX((tid << 10) + 1023)];
		s[0][INDEX((tid << 10) + 1023)] += temp;
		temp = s[1][INDEX((tid << 10) + 511)];
		s[1][INDEX((tid << 10) + 511)] = s[1][INDEX((tid << 10) + 1023)];
		s[1][INDEX((tid << 10) + 1023)] += temp;
		temp = s[2][INDEX((tid << 10) + 511)];
		s[2][INDEX((tid << 10) + 511)] = s[2][INDEX((tid << 10) + 1023)];
		s[2][INDEX((tid << 10) + 1023)] += temp;
		temp = s[3][INDEX((tid << 10) + 511)];
		s[3][INDEX((tid << 10) + 511)] = s[3][INDEX((tid << 10) + 1023)];
		s[3][INDEX((tid << 10) + 1023)] += temp;
	}
	if (tid < 4) {
		temp = s[0][INDEX((tid << 9) + 255)];
		s[0][INDEX((tid << 9) + 255)] = s[0][INDEX((tid << 9) + 511)];
		s[0][INDEX((tid << 9) + 511)] += temp;
		temp = s[1][INDEX((tid << 9) + 255)];
		s[1][INDEX((tid << 9) + 255)] = s[1][INDEX((tid << 9) + 511)];
		s[1][INDEX((tid << 9) + 511)] += temp;
		temp = s[2][INDEX((tid << 9) + 255)];
		s[2][INDEX((tid << 9) + 255)] = s[2][INDEX((tid << 9) + 511)];
		s[2][INDEX((tid << 9) + 511)] += temp;
		temp = s[3][INDEX((tid << 9) + 255)];
		s[3][INDEX((tid << 9) + 255)] = s[3][INDEX((tid << 9) + 511)];
		s[3][INDEX((tid << 9) + 511)] += temp;
	}
	if (tid < 8) {
		temp = s[0][INDEX((tid << 8) + 127)];
		s[0][INDEX((tid << 8) + 127)] = s[0][INDEX((tid << 8) + 255)];
		s[0][INDEX((tid << 8) + 255)] += temp;
		temp = s[1][INDEX((tid << 8) + 127)];
		s[1][INDEX((tid << 8) + 127)] = s[1][INDEX((tid << 8) + 255)];
		s[1][INDEX((tid << 8) + 255)] += temp;
		temp = s[2][INDEX((tid << 8) + 127)];
		s[2][INDEX((tid << 8) + 127)] = s[2][INDEX((tid << 8) + 255)];
		s[2][INDEX((tid << 8) + 255)] += temp;
		temp = s[3][INDEX((tid << 8) + 127)];
		s[3][INDEX((tid << 8) + 127)] = s[3][INDEX((tid << 8) + 255)];
		s[3][INDEX((tid << 8) + 255)] += temp;
	}
	if (tid < 16) {
		temp = s[0][INDEX((tid << 7) + 63)];
		s[0][INDEX((tid << 7) + 63)] = s[0][INDEX((tid << 7) + 127)];
		s[0][INDEX((tid << 7) + 127)] += temp;
		temp = s[1][INDEX((tid << 7) + 63)];
		s[1][INDEX((tid << 7) + 63)] = s[1][INDEX((tid << 7) + 127)];
		s[1][INDEX((tid << 7) + 127)] += temp;
		temp = s[2][INDEX((tid << 7) + 63)];
		s[2][INDEX((tid << 7) + 63)] = s[2][INDEX((tid << 7) + 127)];
		s[2][INDEX((tid << 7) + 127)] += temp;
		temp = s[3][INDEX((tid << 7) + 63)];
		s[3][INDEX((tid << 7) + 63)] = s[3][INDEX((tid << 7) + 127)];
		s[3][INDEX((tid << 7) + 127)] += temp;
	}
	if (tid < 32) {
		temp = s[0][INDEX((tid << 6) + 31)];
		s[0][INDEX((tid << 6) + 31)] = s[0][INDEX((tid << 6) + 63)];
		s[0][INDEX((tid << 6) + 63)] += temp;
		temp = s[1][INDEX((tid << 6) + 31)];
		s[1][INDEX((tid << 6) + 31)] = s[1][INDEX((tid << 6) + 63)];
		s[1][INDEX((tid << 6) + 63)] += temp;
		temp = s[2][INDEX((tid << 6) + 31)];
		s[2][INDEX((tid << 6) + 31)] = s[2][INDEX((tid << 6) + 63)];
		s[2][INDEX((tid << 6) + 63)] += temp;
		temp = s[3][INDEX((tid << 6) + 31)];
		s[3][INDEX((tid << 6) + 31)] = s[3][INDEX((tid << 6) + 63)];
		s[3][INDEX((tid << 6) + 63)] += temp;
	}
	__syncthreads();
	if (tid < 64) {
		temp = s[0][INDEX((tid << 5) + 15)];
		s[0][INDEX((tid << 5) + 15)] = s[0][INDEX((tid << 5) + 31)];
		s[0][INDEX((tid << 5) + 31)] += temp;
		temp = s[1][INDEX((tid << 5) + 15)];
		s[1][INDEX((tid << 5) + 15)] = s[1][INDEX((tid << 5) + 31)];
		s[1][INDEX((tid << 5) + 31)] += temp;
		temp = s[2][INDEX((tid << 5) + 15)];
		s[2][INDEX((tid << 5) + 15)] = s[2][INDEX((tid << 5) + 31)];
		s[2][INDEX((tid << 5) + 31)] += temp;
		temp = s[3][INDEX((tid << 5) + 15)];
		s[3][INDEX((tid << 5) + 15)] = s[3][INDEX((tid << 5) + 31)];
		s[3][INDEX((tid << 5) + 31)] += temp;
	}
	__syncthreads();
	if (tid < 128) {
		temp = s[0][INDEX((tid << 4) + 7)];
		s[0][INDEX((tid << 4) + 7)] = s[0][INDEX((tid << 4) + 15)];
		s[0][INDEX((tid << 4) + 15)] += temp;
		temp = s[1][INDEX((tid << 4) + 7)];
		s[1][INDEX((tid << 4) + 7)] = s[1][INDEX((tid << 4) + 15)];
		s[1][INDEX((tid << 4) + 15)] += temp;
		temp = s[2][INDEX((tid << 4) + 7)];
		s[2][INDEX((tid << 4) + 7)] = s[2][INDEX((tid << 4) + 15)];
		s[2][INDEX((tid << 4) + 15)] += temp;
		temp = s[3][INDEX((tid << 4) + 7)];
		s[3][INDEX((tid << 4) + 7)] = s[3][INDEX((tid << 4) + 15)];
		s[3][INDEX((tid << 4) + 15)] += temp;
	}
	__syncthreads();
	if (tid < 256) {
		temp = s[0][INDEX((tid << 3) + 3)];
		s[0][INDEX((tid << 3) + 3)] = s[0][INDEX((tid << 3) + 7)];
		s[0][INDEX((tid << 3) + 7)] += temp;
		temp = s[1][INDEX((tid << 3) + 3)];
		s[1][INDEX((tid << 3) + 3)] = s[1][INDEX((tid << 3) + 7)];
		s[1][INDEX((tid << 3) + 7)] += temp;
		temp = s[2][INDEX((tid << 3) + 3)];
		s[2][INDEX((tid << 3) + 3)] = s[2][INDEX((tid << 3) + 7)];
		s[2][INDEX((tid << 3) + 7)] += temp;
		temp = s[3][INDEX((tid << 3) + 3)];
		s[3][INDEX((tid << 3) + 3)] = s[3][INDEX((tid << 3) + 7)];
		s[3][INDEX((tid << 3) + 7)] += temp;
	}
	__syncthreads();
	if (tid < 512) {
		temp = s[0][INDEX((tid << 2) + 1)];
		s[0][INDEX((tid << 2) + 1)] = s[0][INDEX((tid << 2) + 3)];
		s[0][INDEX((tid << 2) + 3)] += temp;
		temp = s[1][INDEX((tid << 2) + 1)];
		s[1][INDEX((tid << 2) + 1)] = s[1][INDEX((tid << 2) + 3)];
		s[1][INDEX((tid << 2) + 3)] += temp;
		temp = s[2][INDEX((tid << 2) + 1)];
		s[2][INDEX((tid << 2) + 1)] = s[2][INDEX((tid << 2) + 3)];
		s[2][INDEX((tid << 2) + 3)] += temp;
		temp = s[3][INDEX((tid << 2) + 1)];
		s[3][INDEX((tid << 2) + 1)] = s[3][INDEX((tid << 2) + 3)];
		s[3][INDEX((tid << 2) + 3)] += temp;
	}
	__syncthreads();
	if (tid < 1024) {
		temp = s[0][INDEX((tid << 1))];
		s[0][INDEX((tid << 1))] = s[0][INDEX((tid << 1) + 1)];
		s[0][INDEX((tid << 1) + 1)] += temp;
		temp = s[1][INDEX((tid << 1))];
		s[1][INDEX((tid << 1))] = s[1][INDEX((tid << 1) + 1)];
		s[1][INDEX((tid << 1) + 1)] += temp;
		temp = s[2][INDEX((tid << 1))];
		s[2][INDEX((tid << 1))] = s[2][INDEX((tid << 1) + 1)];
		s[2][INDEX((tid << 1) + 1)] += temp;
		temp = s[3][INDEX((tid << 1))];
		s[3][INDEX((tid << 1))] = s[3][INDEX((tid << 1) + 1)];
		s[3][INDEX((tid << 1) + 1)] += temp;
	}
	__syncthreads();
	

	if (tid == BLOCK_SIZE - 1) {
		if (!isLast) {
			prefixSum_buffer[0][out_offset + blockIdx.x] = s[0][INDEX(LAYER_SIZE - 1)] + last_0;
			prefixSum_buffer[1][out_offset + blockIdx.x] = s[1][INDEX(LAYER_SIZE - 1)] + last_1;
			prefixSum_buffer[2][out_offset + blockIdx.x] = s[2][INDEX(LAYER_SIZE - 1)] + last_2;
			prefixSum_buffer[3][out_offset + blockIdx.x] = s[3][INDEX(LAYER_SIZE - 1)] + last_3;
		} else {
			histgram_buffer[1] = s[0][INDEX(LAYER_SIZE - 1)] + last_0;
			histgram_buffer[2] = s[0][INDEX(LAYER_SIZE - 1)] + last_0 + s[1][INDEX(LAYER_SIZE - 1)] + last_1;
			histgram_buffer[3] = s[0][INDEX(LAYER_SIZE - 1)] + last_0 + s[1][INDEX(LAYER_SIZE - 1)] + last_1 + s[2][INDEX(LAYER_SIZE - 1)] + last_2;
		}
	}

	prefixSum_buffer[0][ai + bid + in_offset] = s[0][INDEX(ai)];
	prefixSum_buffer[0][bi + bid + in_offset] = s[0][INDEX(bi)];
	prefixSum_buffer[1][ai + bid + in_offset] = s[1][INDEX(ai)];
	prefixSum_buffer[1][bi + bid + in_offset] = s[1][INDEX(bi)];
	prefixSum_buffer[2][ai + bid + in_offset] = s[2][INDEX(ai)];
	prefixSum_buffer[2][bi + bid + in_offset] = s[2][INDEX(bi)];
	prefixSum_buffer[3][ai + bid + in_offset] = s[3][INDEX(ai)];
	prefixSum_buffer[3][bi + bid + in_offset] = s[3][INDEX(bi)];
}

}

class RadixSorterv7 :public Sorter
{
	// Layerwise Prefix sum with Work-Efficient Sum Scan
	// Avoid Bank Conflict
	// Unroll
	// Improve Reorder
public:
	
	void sort(DEBUG_FLOAT*& datas , int data_length) override{
		
		// Padding 
		int n = ceil(data_length, LAYER_SIZE) * LAYER_SIZE;
		int pad_len = n - data_length;

		// Init memory
		DEBUG_FLOAT *dev_data[2] = {0};		
		unsigned int *dev_histgram = 0;
		unsigned int *dev_layer_offset = 0;
		unsigned int *dev_prefixSumMem = 0;
		unsigned int **dev_prefixSum = 0;
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


		unsigned int *prefixSumPointer[BIT_RANGE];
		cudaMalloc((void**)&dev_histgram, BIT_RANGE * sizeof(int));
		cudaMalloc((void**)&dev_prefixSum, BIT_RANGE * sizeof(int*));
		cudaMalloc((void**)&dev_prefixSumMem, BIT_RANGE * prefix_sum_size * sizeof(int));
		for (int i = 0; i < BIT_RANGE; ++i) {
			prefixSumPointer[i] = &dev_prefixSumMem[i * prefix_sum_size];
		}
		cudaMemcpy(dev_prefixSum, prefixSumPointer, BIT_RANGE * sizeof(int*), cudaMemcpyHostToDevice);
		// Initialize the buffer
		cudaMalloc((void**)&dev_layer_offset, (num_layer + 1) * sizeof(int));
		cudaMemcpy(dev_layer_offset, layer_offset, (num_layer + 1) * sizeof(int), cudaMemcpyHostToDevice);


		// Kernel launch
		int numBlocks = ceil(n, BLOCK_SIZE);
		int numLayerBlocks = ceil(n, LAYER_SIZE);

		// Prepare index and sortable conversion
		v7::preprocess_float<<< numBlocks , BLOCK_SIZE >>>(dev_data[pout]);
		
		for(int i = 0; i < BITS_LEN; ++i) {
			pout = 1 - pout; // swap double buffer indices
			pin = 1 - pout;

			cudaMemset(dev_histgram , 0, BIT_RANGE * sizeof(int));
			cudaMemset(dev_prefixSumMem, 0, BIT_RANGE * prefix_sum_size * sizeof(int));
			cudaDeviceSynchronize();

			v7::init_sort<<< numBlocks , BLOCK_SIZE >>>(dev_data[pin],
														dev_prefixSum,
														i);
			
			// Prefix_sum
			int num_prefix_sum_block = n;		
			
			for (int layer = 0; layer < num_layer; ++layer) {
				num_prefix_sum_block = ceil(num_prefix_sum_block, LAYER_SIZE);
				v7::block_prefix_sum<<< num_prefix_sum_block, BLOCK_SIZE >>>(dev_prefixSum, 
																		layer_offset[layer], 
																		layer_offset[layer + 1], 
																		dev_histgram,
																		layer == num_layer - 1,
																		i);
			}
			// Sort
			v7::reorder<<< numLayerBlocks , BLOCK_SIZE >>>(dev_data[pout],
													  dev_data[pin], 
													  n, 
													  dev_prefixSum, 
													  dev_histgram,
													  num_layer,
													  dev_layer_offset, 
													  i);
			
		}
		
		
		// Copy origin_datas into sortable by index.
		v7::postprocess_float<<< numBlocks , BLOCK_SIZE >>>(dev_data[pout]);

		// Clear kernel
		cudaMemcpy(datas, dev_data[pout] + pad_len, data_length * sizeof(float), cudaMemcpyDeviceToHost);
		
		cudaFree(dev_data[pout]);
		cudaFree(dev_data[pin]);
		cudaFree(dev_prefixSumMem);
		cudaFree(dev_prefixSum);
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