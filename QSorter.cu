#include <stdio.h>
#include <stdlib.h>
#include "Sorter.h"

class QuickSorter :public Sorter
{
public:
	//QuickSorter :public Sorter();
	//~QuickSorter :public Sorter();
	void sort(DEBUG_FLOAT*& datas , int data_length) override{
		qsort(datas, data_length, sizeof(float), compare);
	}
private:
	static int compare(const void* a, const void* b) {
		auto c = (*(float*)a - *(float*)b);
		return c>0;
	}
	/*
	// compare function including -0.0 < 0.0 detection
	static int compare(const void* a, const void* b) {

		auto c = (*(float*)a - *(float*)b);

		int int_a = *reinterpret_cast<const int*>(a) >> 31;
		int int_b = *reinterpret_cast<const int*>(b) >> 31;

		if (c == 0 && (int_a != int_b)) {
			int int_a = *reinterpret_cast<const int*>(a);
			int int_b = *reinterpret_cast<const int*>(b);
			return (int_a > int_b) - (int_a < int_b);
		}
		return c > 0;

	}
	*/
};
