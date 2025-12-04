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

// Function declarations
std::vector<float> read_csv(const std::string& path, int height, int width);
void write_csv(const std::vector<float>& data, const std::string& path);

#endif // UTILS_HPP
