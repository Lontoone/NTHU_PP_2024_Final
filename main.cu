#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <iostream>

// Our Sorters....
#include "QSorter.cu"
#include "radix_sort_slow.cu"

// For reading file input
#include <string.h>
#include <fstream>
#include <sstream>
#include <filesystem>
// for timing
#include <chrono>

std::unique_ptr<Sorter> get_sorter(std::string input) {
	if (input == "qsort") {
		return std::make_unique<QuickSorter>();
	}
	else if (input == "radixsort") {
		//.... return ours sorter
		return std::make_unique<QuickSorter>();
	}
	else if(input =="slr"){
		return std::make_unique<SlowRadixSorter>();
	}

	throw std::invalid_argument("unknow sorter type");

}

bool compare_answer(float* data1 , float* data2 , int data_length) {
	bool is_the_same = true;
	for (int i = 0; i < data_length; ++i) {
		
		if (data1[i] != data2[i]) {
			return false;
		}
	}
	return true;
}

bool input_data(DEBUG_FLOAT*& datas , int& data_length ,const char* path) {
	std::ifstream file(path); // Open the file
	if (!file.is_open()) {
		std::cerr << "Failed to open the file." << std::endl;
		return 1;
	}

	std::string line;
	int i = 0;
	while (std::getline(file, line)) { // Read each line
		if (i == 0)
		{
			std::stringstream ss(line);			
			ss >> data_length;
			// first element is data length
			datas = (DEBUG_FLOAT*)malloc(data_length *sizeof(DEBUG_FLOAT));
		}
		else {
			std::stringstream ss(line);
			DEBUG_FLOAT value;
			ss >> value; // Convert line to float
			if (ss.fail()) {
				std::cerr << "Failed to convert line to float: " << line << std::endl;
				continue;
			}
			datas[i-1] = value; // Store the float in the vector
		}

		++i;
	}

	file.close(); 

}

int main(int argc, char* argv[]) {
		
	std::string sort_method = "qsort";
	std::string data_path = "../../data/c01.txt";
	// Parse command-line arguments
	if (argc > 1) {
		printf("argc %d \n" ,argc);
		data_path = argv[1];

		for (int i = 2; i < argc; ++i) {
			if (std::string(argv[i]) == "-s" && i + 1 < argc) { // for sort argment
				sort_method = argv[i + 1];
				++i; // Skip the next argument as it is the value for -s
			}
		}
	}

	//==============================
	//		Compare mode
	//==============================	
	printf("data path %s \n", data_path.c_str());
	printf("sort method %s \n" , sort_method.c_str());

	// Load data	
	int data_length = 0;
	DEBUG_FLOAT* datas= NULL;
	input_data(datas,data_length , data_path.c_str());
	// Copy data for gt_sort to perform on
	DEBUG_FLOAT* GT_datas = (DEBUG_FLOAT*)malloc(data_length*sizeof(DEBUG_FLOAT));
	std::copy(datas , datas + data_length , GT_datas);

	// Get sorter by input
	auto GT_sorter = get_sorter("qsort");
	auto ours_sorter = get_sorter(sort_method);

	auto start = std::chrono::high_resolution_clock::now();
	ours_sorter->sort(datas,data_length);
    auto end = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> duration = end - start;
	printf("Your sort code runs for : %f secondes \n" , duration.count());

	// Compare with GT
	GT_sorter->sort(GT_datas,data_length);

	// Debug
	/* 
	*/
	printf("data_length %d \n" , data_length);
	for (int i = 0; i < data_length; ++i) {
		printf("%f vs %f \n" , datas[i] , GT_datas[i]);
	} 
	bool is_correct = compare_answer(GT_datas , datas , data_length);

	if (is_correct) {
		printf("Correct! \n" );
	}
	else {
		printf("Inorrect :( \n");
	}
	/*
	*/

	system("pause");
	free(datas);
	free(GT_datas);

}  