#include <iostream>
#include <string.h>
#include <random>
#include <fstream>

int main(int argc, char* argv[]) {
	std::string output_path = "../../data/c99.txt";
	std::string gen_method ="random";
	int data_length =0;
	if(argc>1){
		output_path = argv[1];
		data_length= std::atoi( argv[2]);
		gen_method = argv[3];
	}


	if(gen_method=="random"){
		// Create a random number generator
		std::random_device rd; // Seed
		std::mt19937 gen(rd()); // Mersenne Twister engine
		std::uniform_real_distribution<> dis(-99999.0, 99999.0); // Uniform 

		// Open a file to write the random numbers
		std::ofstream outFile(output_path);
		if (!outFile.is_open()) {
			std::cerr << "Failed to open the file." << std::endl;
			return 1;
		}

		// Generate and write random numbers to the file
		outFile << data_length << std::endl;
		for (int i = 0; i < data_length; ++i) {
			float randomNumber = dis(gen);
			outFile << randomNumber << std::endl;
		}

		outFile.close();

	}



}