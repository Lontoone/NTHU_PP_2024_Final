# How to start

```
mkdir build
cd build
cmake ../
cmake --build .
```

You will build 2 exe file : `GenData.exe` and `Final_Project.exe`. One for data generation one for sorter.

# Main sorter:

Run this command :
```
Final_Project.exe Test_case_path -s Sort_method
```

Example:
```
Final_Project.exe ../../data/c02.txt -s qsort
```

## 如何做你的Sorter

參考Qsort.cu

```
class YourSorter :public Sorter
{
public:
	void sort(float* datas , int data_length) override{
		//... Your sorter algo...
	}
}
```

在 main.cu加上get_sorter
```
std::unique_ptr<Sorter> get_sorter(std::string input) {
	if (input == "qsort") {
		return std::make_unique<QuickSorter>();
	}
	else if (input == "radixsort") {
		//.... return ours sorter
		return std::make_unique<YourSorter>();
	}
	throw std::invalid_argument("unknow sorter type");

}
```


# Generate data

Run this command:
```
GenData.exe output_path number method
```

Example:
```
GenData.exe c02.txt 90 random
```