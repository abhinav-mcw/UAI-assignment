#include <iostream>
#include <vector>
#include <fstream>
#include <stdexcept>
#include <cstring>
#include <sstream>
#include <cnpy.h>

// template <typename T>
std::vector<std::vector<std::vector<std::vector<float>>>> read_npy_file(
            const std::string &filename,
            std::vector<int> &shape);
void write_to_binary(const std::string &filename, const std::vector<float> &data);