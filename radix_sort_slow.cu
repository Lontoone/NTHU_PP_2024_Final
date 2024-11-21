#include <stdio.h>
#include <stdlib.h>
#include "Sorter.h"

#define THREAD_W 16
#define BITS_LEN 32

__device__ void extractBits(unsigned int d_num, int numBits) {
    for(int i = BITS_LEN -1 ; i>0 ; --i)
	{
        unsigned int mask = 1 << (i);
        unsigned int a =   (d_num & mask) ? 1 : 0;
		printf(" %d" ,a  );
    }
	printf("\n");
}

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
	printf("idx %d , n %d \n " , idx , n);
	/*
	// TEMP : Debug mode do not need float process
	float f = data[idx];
    uint32_t fu = __float_as_uint(f);
    uint32_t mask = __float_as_uint(-(int32_t(fu) >> 31)) | 0x80000000u;

	index_buffers[idx] = idx;
	data[idx] = fu ^ mask;	
	*/

}
__global__ void prefix_sum_hist(
	DEBUG_FLOAT* data ,
	unsigned int* prefixSum_0_buffer,	
	unsigned int* histgram_buffer ,
	int n,
	int bit
	){
		uint32_t bit_data = (uint32_t(data[n-1]) & (1<<bit))>>bit;
		bool is_one = bit_data & 1 ;
		histgram_buffer[ 1 ] = prefixSum_0_buffer[n-1] + !is_one;
		
		printf("     %d  %d \n" , histgram_buffer[ 1 ] , histgram_buffer[ 0 ]);
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

	if(idx >= BITS_LEN){
		return;
	}

	for(int i=0 ; i<n-1;++i){
		//int bit = idx;		
		uint32_t bit_data = (uint32_t(data[i]) & (1<<bit))>>bit;
		bool is_one = bit_data & 1 ;
		//printf("bitdata %d bit %d  idx : %d :  %d \n" ,bit_data,(bit_data & 1) ,bit * n + i , prefixSum_buffer[bit * n + i+1 ] );
		prefixSum_1_buffer[ i +1 ] = prefixSum_1_buffer[ i ] + is_one;	
		prefixSum_0_buffer[ i +1 ] = prefixSum_0_buffer[ i ] + !is_one;	
		/*
		if(!is_one){
			atomicAdd(&histgram_buffer[ 1 ] ,1);
		}
		else{
			// prefix sum for 0 is always 0
			//atomicAdd(&histgram_buffer[bit * 2 + 0 ] ,1);
		}
		*/
	}
	
	/*
	for(int i=0 ; i<n-1;++i){
		int bit = idx;		
		uint32_t bit_data = (uint32_t(data[i]) & (1<<bit))>>bit;
		bool is_one = bit_data & 1 ;
		//printf("bitdata %d bit %d  idx : %d :  %d \n" ,bit_data,(bit_data & 1) ,bit * n + i , prefixSum_buffer[bit * n + i+1 ] );
		prefixSum_1_buffer[bit * n + i +1 ] = prefixSum_1_buffer[bit * n + i ] + is_one;	
		prefixSum_0_buffer[bit * n + i +1 ] = prefixSum_0_buffer[bit * n + i ] + !is_one;	

		if(!is_one){
			atomicAdd(&histgram_buffer[bit * 2 + 1 ] ,1);
		}
		else{
			// prefix sum for 0 is always 0
			//atomicAdd(&histgram_buffer[bit * 2 + 0 ] ,1);
		}
	}
	*/
	__syncthreads();
	if(idx == 0){
		printf("\n idx | 1 | 0  \n");
		for(int i = 0 ; i < n ; i++){		
			printf(" %d :  %d  %d \n" , i, prefixSum_1_buffer[idx * n + i]  , prefixSum_0_buffer[idx * n + i] );

		}

	}

	/*
	// For each number, calculate its bit.
	for(int bit=0;bit<BITS_LEN;bit++){
		//uint32_t bit_data = (__float_as_uint(data[idx]) & (1<<bit))>>bit; // TODO: This may not work
		uint32_t bit_data = (uint32_t(data[idx]) & (1<<bit))>>bit;
		//bool is_one = bit_data & 1 ;

		prefixSum_buffer[bit * n + idx +1 ] = prefixSum_buffer[bit * n + idx ] + (bit_data & 1);
		__syncthreads();

		if(bit == 0){
			printf("bitdata %d bit %d  idx : %d :  %d \n" ,__float_as_uint(data[idx]),(bit_data & 1) ,bit * n + idx , prefixSum_buffer[bit * n + idx+1 ] );

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

	uint32_t bit_data = (uint32_t(data[idx]) & (1<<sort_bit))>>sort_bit;
	bool is_one = bit_data & 1 ;
	if(is_one){
		index_buffers[idx ] = histgram_buffer[1] + prefixSum_1_buffer[idx];
	}
	else{
		index_buffers[idx ] = histgram_buffer[0] + prefixSum_0_buffer[idx];
	}
	data[index_buffers[idx ]] = data_origin[idx];
	printf(" %d ",index_buffers[idx ]);

}
__global__ void postprocess_float(float* const sort_able_data , float* const origin_data,int n ,  unsigned int* index_buffers){
	int globalThreadIdX  = blockIdx.x * blockDim.x + threadIdx.x;
    int globalThreadIdY  = blockIdx.y * blockDim.y + threadIdx.y;
	int idx  = globalThreadIdY * (gridDim.x * THREAD_W) + globalThreadIdX; 

	if(idx >=n){
		return;
	}
	unsigned int index = index_buffers[idx];
	sort_able_data[idx] = origin_data[index];

	/*
	// Assuming fu is the transformed value
	uint32_t fu = __float_as_uint(origin_data[idx]);
	uint32_t mask = (((fu >> 31)-1) | 0x80000000u);

	// Reverse the transformation
	uint32_t original_fu = fu ^ mask;
	float original_f = __uint_as_float(original_fu);

	// Store the original float back to data
	data[idx] = original_f;
    //return fu ^ mask;
	*/
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

	prefixSum_0_buffer[idx] = 0;
	prefixSum_1_buffer[idx] = 0;

	if(idx ==0){		
		histgram_buffer[1] = 0;
	}
	
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

		for(int i = 0 ; i< 4 ; i++){
			init_sort<<< numBlocks , threadsPerBlock >>>(dev_1prefixSum , dev_0prefixSum , dev_histgram , data_length , dev_datas_sortable , dev_origin_datas);
			printf("\n");
			// Prefix_sum				
			prefix_sum<<< 1 , 1 >>>(dev_datas_sortable, data_length , dev_1prefixSum , dev_0prefixSum , dev_histgram , i);
			prefix_sum_hist<<<1,1>>>(dev_datas_sortable , dev_0prefixSum , dev_histgram , data_length ,i );
			// Sort
			reorder<<< numBlocks , threadsPerBlock >>>(dev_datas_sortable , dev_origin_datas, data_length , dev_1prefixSum , dev_0prefixSum , dev_histgram , dev_index_buffer, i);
			printf("\n");
			printf("\n");
			cudaMemcpy(datas, dev_datas_sortable, data_length * sizeof(float), cudaMemcpyDeviceToHost);
			printf("\n============================= datas =================================\n");
			for(int i = 0 ; i < data_length ; i++){
				printBits(datas[i]);			
			}
			printf("\n");
		}
		// Copy origin_datas into sortable by index.
		//postprocess_float<<< numBlocks , threadsPerBlock >>>(dev_datas_sortable , dev_origin_datas, data_length , dev_index_buffer);
		

		// Clear kernel
		cudaMemcpy(datas, dev_datas_sortable, data_length * sizeof(float), cudaMemcpyDeviceToHost);
		
		cudaFree(dev_datas_sortable);
		cudaFree(dev_origin_datas);
		cudaFree(dev_index_buffer);
		cudaFree(dev_0prefixSum);
		cudaFree(dev_1prefixSum);
		cudaFree(dev_histgram);

		/*
		printf("============================= datas =================================\n");
		for(int i = 0 ; i < data_length ; i++){
			printBits(datas[i]);			
		}
		printf("\n");

		printf("============================= prefix sum =================================\n");
		for(int i = 0 ; i < data_length ; i++){
			//printf("%d " ,  );
		}
		*/
	}
private:
	void printBits(unsigned int num) {
    int bits = sizeof(num) * 8; // Number of bits in the integer
    for (int i = bits - 1; i >= 0; i--) {
        unsigned int mask = 1 << i;
        printf("%d", (num & mask) ? 1 : 0);
    }
    printf("\n");
}
};
