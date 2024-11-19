#include <stdio.h>
#include <stdlib.h>
#include "Sorter.h"

class SlowRadixSorter :public Sorter
{
public:
	
	void sort(float* datas , int data_length) override{
		int *dev_datas = 0;
	    cudaMalloc((void**)&dev_datas, data_length * sizeof(float));
		cudaMemcpy(dev_datas, datas, data_length * sizeof(float), cudaMemcpyHostToDevice);
		// Init memory


		// Kernel launch
		// Preprocess float 
		// Post Process float

		// Clear kernel
		cudaMemcpy(datas, dev_datas, data_length * sizeof(float), cudaMemcpyHostToDevice);
		cudaFree(dev_datas);
	}
private:
	
};
