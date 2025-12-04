#ifndef RADIOACTIVE_MPI_HPP
#define RADIOACTIVE_MPI_HPP
#include <vector>
#include <cmath>
#include <algorithm>
#include <mpi.h>
#include <iostream>
#include <vector>
#include <cstring>

constexpr int H = 4000;
constexpr int W = 4000;

constexpr float UX = 3.3f;
constexpr float UY = 1.4f;

constexpr float DIFF_D = 1000.0f;
constexpr float LAMBDA_DECAY = 3e-5f;
constexpr float K_DEP = 1e-4f;

constexpr float DX = 10.0f;
constexpr float DY = 10.0f;

constexpr float DT = 0.01f;

inline int idx(int i, int j) {
    return i * W + j;
}

float calc_next_c(float C, float C_left, float C_right, float C_up, float C_down);
void run_radioactive_mpi_sync(std::vector<float>& radioactive_grid, int steps);

#endif // RADIOACTIVE_MPI_HPP