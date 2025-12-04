#ifndef UTILS_HPP
#define UTILS_HPP

#include <vector>
#include <string>

// Function declarations
std::vector<float> read_csv(const std::string& path, int height, int width);
void write_csv(const std::vector<float>& data, const std::string& path);

#endif // UTILS_HPP
