#pragma once
#define DEBUG_FLOAT float
//#define DEBUG_FLOAT unsigned int
class Sorter
{
public:
		
	virtual void sort(DEBUG_FLOAT*& datas, int data_len) {};
	//Sorter();
	~Sorter() = default;

};

