#include <stdio.h>
#include <stdlib.h>
#include "Sorter.h"
#include <bitset>
#include <iostream>

#define BITS_LEN 32
#define BLOCK_SIZE 1024

namespace v1 {

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

__global__ void prefix_sum_hist(DEBUG_FLOAT* data, unsigned int* prefixSum_0_buffer, unsigned int* histgram_buffer,	int n, int bit) {
	bool is_one = (__float_as_uint(data[n-1]) >> bit) & 1 ;
	histgram_buffer[1] = prefixSum_0_buffer[n-1] + !is_one;
}

__global__ void prefix_sum(DEBUG_FLOAT* data, int n, unsigned int* prefixSum_1_buffer, unsigned int* prefixSum_0_buffer, unsigned int bit) {
	// Totoal data length = n(欄) * bit_len(row) * 2 (0 or 1)
	// Each thread process each **row**
	for(int i = 0; i < n - 1; ++i){				
		bool is_one = (__float_as_uint(data[i]) >> bit) & 1 ;
		prefixSum_1_buffer[i + 1] = prefixSum_1_buffer[i] + is_one;	
		prefixSum_0_buffer[i + 1] = prefixSum_0_buffer[i] + !is_one;
	}
}

__global__ void reorder(
	DEBUG_FLOAT* data , 
	DEBUG_FLOAT* data_origin , 
	int n , 
	unsigned int* prefixSum_1_buffer,
	unsigned int* prefixSum_0_buffer,
	unsigned int* histgram_buffer,
	int sort_bit
	)
{
	int idx  = blockIdx.x * blockDim.x + threadIdx.x;

	// uint32_t bit_data = (__float_as_uint(data_origin[idx]) & (1<<sort_bit))>>sort_bit;
	// uint32_t bit_data = (uint32_t(data[idx]) & (1<<sort_bit))>>sort_bit;
	bool is_one = (__float_as_uint(data_origin[idx]) >> sort_bit) & 1;
	int reordered_idx = 0;

	
	if(is_one) reordered_idx = histgram_buffer[1] + prefixSum_1_buffer[idx];
	else reordered_idx = prefixSum_0_buffer[idx];
					
	data[reordered_idx] = data_origin[idx];
}



__global__ void init_sort(DEBUG_FLOAT* data_sortable, DEBUG_FLOAT* data_origin) {
	int idx  = blockIdx.x * blockDim.x + threadIdx.x;
	data_origin[idx] = data_sortable[idx];
}

}

class RadixSorterv1 :public Sorter
{
	// Add Padding to maximize threads utilization
	// Remove Branch
	// Modify Pre- and Post- Process
public:
	
	void sort(DEBUG_FLOAT*& datas , int data_length) override{

		// Padding 
		int n = ceil(data_length, BLOCK_SIZE) * BLOCK_SIZE;
		int offset = n - data_length;

		// Init memory
		DEBUG_FLOAT *dev_datas_sortable = 0;		
		DEBUG_FLOAT *dev_origin_datas = 0;	
		unsigned int *dev_histgram = 0;
		unsigned int *dev_1prefixSum = 0;
		unsigned int *dev_0prefixSum = 0;

		// Buffer	    
		cudaMalloc((void**)&dev_datas_sortable, n * sizeof(DEBUG_FLOAT));
		cudaMemset(dev_datas_sortable + data_length, 0xff,  offset * sizeof(DEBUG_FLOAT));
		cudaMemcpy(dev_datas_sortable, datas, data_length * sizeof(DEBUG_FLOAT), cudaMemcpyHostToDevice);

	    cudaMalloc((void**)&dev_origin_datas, n * sizeof(DEBUG_FLOAT));
		cudaMemset(dev_origin_datas + data_length, 0xff,  offset * sizeof(DEBUG_FLOAT));
		cudaMemcpy(dev_origin_datas, datas, data_length * sizeof(DEBUG_FLOAT), cudaMemcpyHostToDevice);

		// Histgram and prefix sum
		cudaMalloc((void**)&dev_histgram, 2 * sizeof(int));
		cudaMalloc((void**)&dev_1prefixSum, n * sizeof(int));
		cudaMalloc((void**)&dev_0prefixSum, n * sizeof(int));
		
		// Initialize the buffer with zeros
		cudaMemset(dev_histgram , 0, 2 * sizeof(int));
		cudaMemset(dev_1prefixSum, 0, n * sizeof(int));
		cudaMemset(dev_0prefixSum, 0, n * sizeof(int));

		// Kernel launch
		int numBlocks = ceil(n, BLOCK_SIZE);

		// Prepare index and sortable conversion
		v1::preprocess_float<<< numBlocks , BLOCK_SIZE >>>(dev_datas_sortable);
		
		for(int i = 0; i < BITS_LEN; ++i) {
			v1::init_sort<<< numBlocks , BLOCK_SIZE >>>(dev_datas_sortable, dev_origin_datas);
			cudaMemset(dev_histgram , 0,  2 * sizeof(int));
			cudaMemset(dev_1prefixSum, 0,  n * sizeof(int));
			cudaMemset(dev_0prefixSum, 0,  n * sizeof(int));
			cudaDeviceSynchronize();
			
			//printf("\n");
			// Prefix_sum				
			v1::prefix_sum<<< 1 , 1 >>>(dev_datas_sortable, n , dev_1prefixSum , dev_0prefixSum, i);
			v1::prefix_sum_hist<<< 1 , 1 >>>(dev_datas_sortable , dev_0prefixSum , dev_histgram , n ,i );
			// Sort
			v1::reorder<<< numBlocks , BLOCK_SIZE >>>(dev_datas_sortable , dev_origin_datas, n , dev_1prefixSum , dev_0prefixSum , dev_histgram, i);
		}
		
		/*
		*/
		
		// Copy origin_datas into sortable by index.
		v1::postprocess_float<<< numBlocks , BLOCK_SIZE >>>(dev_datas_sortable);
		

		// Clear kernel
		cudaMemcpy(datas, dev_datas_sortable + offset, data_length * sizeof(float), cudaMemcpyDeviceToHost);
		
		cudaFree(dev_datas_sortable);
		cudaFree(dev_origin_datas);
		cudaFree(dev_0prefixSum);
		cudaFree(dev_1prefixSum);
		cudaFree(dev_histgram);
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
