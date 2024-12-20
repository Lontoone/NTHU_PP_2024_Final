#include <stdio.h>
#include <stdlib.h>
#include "Sorter.h"
#include <bitset>
#include <iostream>

#define THREAD_W 16
#define BITS_LEN 32

__global__ void preprocess_float(
	DEBUG_FLOAT* data , 
	int n , 
	unsigned int* index_buffers 
	){
	int globalThreadIdX  = blockIdx.x * blockDim.x + threadIdx.x;
    int globalThreadIdY  = blockIdx.y * blockDim.y + threadIdx.y;
	int idx  = globalThreadIdY * (gridDim.x * THREAD_W) + globalThreadIdX; 


	if(idx >=n){
		return;
	}
 	unsigned int *data_temp = (unsigned int *)(&data[idx]);    
    *data_temp = (*data_temp >> 31 & 0x1)? ~(*data_temp): (*data_temp) | 0x80000000; 

}
__global__ void prefix_sum_hist(
	DEBUG_FLOAT* data ,
	unsigned int* prefixSum_0_buffer,	
	unsigned int* histgram_buffer ,
	int n,
	int bit
	){
		uint32_t bit_data = (__float_as_uint(data[n-1]) & (1<<bit))>>bit;
		bool is_one = bit_data & 1 ;
		
		histgram_buffer[ 1 ] = prefixSum_0_buffer[n-1] + !is_one;
		
		//printf("     %d  %d \n" , histgram_buffer[ 1 ] , histgram_buffer[ 0 ]);
	}

__global__ void prefix_sum(
	DEBUG_FLOAT* data , 
	int n , 
	unsigned int* prefixSum_1_buffer,
	unsigned int* prefixSum_0_buffer,
	unsigned int* histgram_buffer , 
	unsigned int bit
	)
	{
	// Totoal data length = n(欄) * bit_len(row) * 2 (0 or 1)
	// Each thread process each **row**

	int globalThreadIdX  = blockIdx.x * blockDim.x + threadIdx.x;
    int globalThreadIdY  = blockIdx.y * blockDim.y + threadIdx.y;
	int idx  = globalThreadIdY * (gridDim.x * THREAD_W) + globalThreadIdX; 
	/*
	if(idx >= BITS_LEN){
		return;
	}
	*/

	for(int i=0 ; i<n-1;++i){		
		uint32_t bit_data = (__float_as_uint(data[i]) & (1<<bit))>>bit;
		
		bool is_one = bit_data & 1 ;
		//printf("bitdata %d bit %d  idx : %d :  %d \n" ,bit_data,(bit_data & 1) ,bit * n + i , prefixSum_buffer[bit * n + i+1 ] );
		prefixSum_1_buffer[ i +1 ] = prefixSum_1_buffer[ i ] + is_one;	
		prefixSum_0_buffer[ i +1 ] = prefixSum_0_buffer[ i ] + !is_one;	
	
	}
	
	
	__syncthreads();
	/*
	if(idx == 0){
		printf("\n idx | 1 | 0  \n");
		for(int i = 0 ; i < n ; i++){		
			printf(" %d :  %d  %d \n" , i, prefixSum_1_buffer[idx * n + i]  , prefixSum_0_buffer[idx * n + i] );

		}

	}
	*/

}
__global__ void reorder(
	DEBUG_FLOAT* data , 
	DEBUG_FLOAT* data_origin , 
	int n , 
	unsigned int* prefixSum_1_buffer,
	unsigned int* prefixSum_0_buffer,
	unsigned int* histgram_buffer,
	unsigned int* index_buffers,
	int sort_bit
	)
{
	int globalThreadIdX  = blockIdx.x * blockDim.x + threadIdx.x;
    int globalThreadIdY  = blockIdx.y * blockDim.y + threadIdx.y;
	int idx  = globalThreadIdY * (gridDim.x * THREAD_W) + globalThreadIdX; 

	if(idx >=n){
		return;
	}

	uint32_t bit_data = (__float_as_uint(data[idx]) & (1<<sort_bit))>>sort_bit;
	//uint32_t bit_data = (uint32_t(data[idx]) & (1<<sort_bit))>>sort_bit;
	bool is_one = bit_data & 1 ;
	if(is_one){
		index_buffers[idx ] = histgram_buffer[1] + prefixSum_1_buffer[idx];
	}
	else{
		index_buffers[idx ] = histgram_buffer[0] + prefixSum_0_buffer[idx];
	}
	data[index_buffers[idx ]] = data_origin[idx];
	//printf(" %d ",index_buffers[idx ]);

}
__global__ void postprocess_float(DEBUG_FLOAT* const sort_able_data , DEBUG_FLOAT* const origin_data,int n ,  unsigned int* index_buffers){
	int globalThreadIdX  = blockIdx.x * blockDim.x + threadIdx.x;
    int globalThreadIdY  = blockIdx.y * blockDim.y + threadIdx.y;
	int idx  = globalThreadIdY * (gridDim.x * THREAD_W) + globalThreadIdX; 

	if(idx >=n){
		return;
	}

	unsigned int* data_temp = (unsigned int *)(&sort_able_data[idx]);
    *data_temp = (*data_temp >> 31 & 0x1)? (*data_temp) & 0x7fffffff: ~(*data_temp);

}

__global__ void init_sort(
	unsigned int* prefixSum_1_buffer,
	unsigned int* prefixSum_0_buffer,
	unsigned int* histgram_buffer,
	int n ,
	DEBUG_FLOAT* data_sortable , 
	DEBUG_FLOAT* data_origin 
	){
	int globalThreadIdX  = blockIdx.x * blockDim.x + threadIdx.x;
    int globalThreadIdY  = blockIdx.y * blockDim.y + threadIdx.y;
	int idx  = globalThreadIdY * (gridDim.x * THREAD_W) + globalThreadIdX; 

	if(idx >=n){
		return;
	}
	/*
	prefixSum_0_buffer[idx] = 0;
	prefixSum_1_buffer[idx] = 0;

	if(idx ==0)
	{		
		histgram_buffer[0] = 0;
		histgram_buffer[1] = 0;
	}
	*/
	
	data_origin[idx] = data_sortable[idx];

}


class SlowRadixSorter :public Sorter
{
public:
	
	void sort(DEBUG_FLOAT*& datas , int data_length) override{

		// Init memory
		DEBUG_FLOAT *dev_datas_sortable = 0;		
		DEBUG_FLOAT *dev_origin_datas = 0;	
		unsigned int *dev_index_buffer = 0;	
		unsigned int *dev_histgram = 0;
		unsigned int *dev_1prefixSum = 0;
		unsigned int *dev_0prefixSum = 0;

		// Buffer
	    cudaMalloc((void**)&dev_index_buffer, data_length * sizeof(unsigned int));
	    cudaMalloc((void**)&dev_datas_sortable, data_length * sizeof(DEBUG_FLOAT));
		cudaMemcpy(dev_datas_sortable, datas, data_length * sizeof(DEBUG_FLOAT), cudaMemcpyHostToDevice);
	    cudaMalloc((void**)&dev_origin_datas, data_length * sizeof(DEBUG_FLOAT));
		cudaMemcpy(dev_origin_datas, datas, data_length * sizeof(DEBUG_FLOAT), cudaMemcpyHostToDevice);

		// Histgram and prefix sum
		cudaMalloc((void**)&dev_histgram, 2 * BITS_LEN * sizeof(int));
		cudaMalloc((void**)&dev_1prefixSum, 2 * BITS_LEN * data_length * sizeof(int));
		cudaMalloc((void**)&dev_0prefixSum, 2 * BITS_LEN * data_length * sizeof(int));
		// Initialize the buffer with zeros
		cudaMemset(dev_histgram , 0,  2 * BITS_LEN * sizeof(int));
		cudaMemset(dev_1prefixSum, 0,  2 * BITS_LEN * data_length * sizeof(int));
		cudaMemset(dev_0prefixSum, 0,  2 * BITS_LEN * data_length * sizeof(int));

		// Kernel launch
		dim3 threadsPerBlock(THREAD_W, THREAD_W);
		dim3 numBlocks((data_length + threadsPerBlock.x - 1) / threadsPerBlock.x,
					(data_length + threadsPerBlock.y - 1) / threadsPerBlock.y);

		// Prepare index and sortable conversion
		preprocess_float<<< numBlocks , threadsPerBlock >>>(dev_datas_sortable , data_length , dev_index_buffer);		
		
		for(int i = 0 ; i< BITS_LEN ; i++)		
		//for(int i = 0 ; i< 2 ; i++)
		//int i = 31;
		{
			init_sort<<< numBlocks , threadsPerBlock >>>(dev_1prefixSum , dev_0prefixSum , dev_histgram , data_length , dev_datas_sortable , dev_origin_datas);
			cudaMemset(dev_histgram , 0,  2 * BITS_LEN * sizeof(int));
			cudaMemset(dev_1prefixSum, 0,  data_length * sizeof(int));
			cudaMemset(dev_0prefixSum, 0,  data_length * sizeof(int));
			cudaDeviceSynchronize();
			
			//printf("\n");
			// Prefix_sum				
			prefix_sum<<< 1 , 1 >>>(dev_datas_sortable, data_length , dev_1prefixSum , dev_0prefixSum , dev_histgram , i);
			prefix_sum_hist<<<1,1>>>(dev_datas_sortable , dev_0prefixSum , dev_histgram , data_length ,i );
			// Sort
			reorder<<< numBlocks , threadsPerBlock >>>(dev_datas_sortable , dev_origin_datas, data_length , dev_1prefixSum , dev_0prefixSum , dev_histgram , dev_index_buffer, i);
			//printf("\n");

			/*
			cudaMemcpy(datas, dev_datas_sortable, data_length * sizeof(float), cudaMemcpyDeviceToHost);
			printf("\n============================= datas =================================\n");
			for(int i = 0 ; i < data_length ; i++){
				printBits(datas[i]);								
    			printf("\n");
			}
			*/
		}
		
		/*
		*/
		
		// Copy origin_datas into sortable by index.
		postprocess_float<<< numBlocks , threadsPerBlock >>>(dev_datas_sortable , dev_origin_datas, data_length , dev_index_buffer);
		

		// Clear kernel
		cudaMemcpy(datas, dev_datas_sortable, data_length * sizeof(float), cudaMemcpyDeviceToHost);
		
		cudaFree(dev_datas_sortable);
		cudaFree(dev_origin_datas);
		cudaFree(dev_index_buffer);
		cudaFree(dev_0prefixSum);
		cudaFree(dev_1prefixSum);
		cudaFree(dev_histgram);
		
		printf("============================= datas =================================\n");
		for(int i = 0 ; i < data_length ; i++){
			printBits(datas[i] );			
			printf(" : %f" , datas[i] );
			
    		printf("\n");
		}
		printf("\n");
		/*

		printf("============================= prefix sum =================================\n");
		for(int i = 0 ; i < data_length ; i++){
			//printf("%d " ,  );
		}
		*/
	}
private:
	/*
	void printBits(unsigned int num) {
		int bits = sizeof(num) * 8; // Number of bits in the integer
		for (int i = bits - 1; i >= 0; i--) {
			unsigned int mask = 1 << i;
			printf("%d", (num & mask) ? 1 : 0);		
    	}
	}
	*/
	void printBits(DEBUG_FLOAT num) {
		  unsigned int bits = *reinterpret_cast<unsigned int*>(&num);

		// Print the bits
		std::bitset<32> bitset(bits);
		std::cout << "Bits of " << num << ": " << bitset << std::endl;
	}
};
