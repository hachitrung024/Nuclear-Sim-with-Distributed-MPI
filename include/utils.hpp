#ifndef UTILS_HPP
#define UTILS_HPP
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include "radioactive_mpi.hpp"
constexpr int H = 4000;
constexpr int W = 4000;
inline int idx(int i, int j) {
    return i * W + j;
}
// Function declarations
std::vector<float> read_csv(const std::string& path, int height, int width);
void write_csv(const std::vector<float>& data, const std::string& path);

#endif // UTILS_HPP
