#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include "Sorter.h"
#define RANGE 16
#define DIGITS 8


class CPURadixSorter :public Sorter
{
public:
	
	void sort(DEBUG_FLOAT*& datas, int data_length) override {
		unsigned int *input = (unsigned int*) malloc (data_length * sizeof(unsigned int));
		unsigned int *output = (unsigned int*) malloc (data_length * sizeof(unsigned int));
		
		for (int i = 0; i < data_length; ++i) {
			unsigned int temp = *(unsigned int*)&datas[i];
			input[i] = (temp >> 31 & 0x1)? ~(temp): (temp) | 0x80000000;
		}

		for (int i = 0; i < DIGITS; ++i) {
			counting_sort(input, output, data_length, i);
			swap(&input, &output);
		}

		for (int i = 0; i < data_length; ++i) {
			unsigned int temp = input[i];
			temp = (temp >> 31 & 0x1)? (temp) & 0x7fffffff: ~(temp);
			datas[i] = *(float*)&temp;
		}
	}
private:

	void counting_sort (unsigned int* input, unsigned int* output, int data_length, int index) {
		int count[RANGE];
		for (int i = 0; i < RANGE; ++i) count[i] = 0;
		
		for (int i = 0; i < data_length; ++i) {
			++count[(input[i] >> (index * 4)) & 0x0000000F];
		}

		int offset[RANGE];
		offset[0] = 0;
		for (int i = 1; i < RANGE; ++i) {
			offset[i] = offset[i - 1] + count[i - 1];
			count[i - 1] = 0;
		}
		count[RANGE - 1] = 0;
		
		for (int i = 0; i < data_length; ++i) {
			unsigned int rem = (input[i] >> (index * 4)) & 0x0000000F;
			output[offset[rem] + count[rem]] = input[i];
			++count[rem];
		}
	}

	void swap(unsigned int *a[], unsigned int *b[]) {
		unsigned int *tmp;
		tmp = *a;
		*a = *b;
		*b = tmp;                                                                                             
	}
};
