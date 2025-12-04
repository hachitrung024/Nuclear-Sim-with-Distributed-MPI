#ifndef HEAT_MPI_HPP
#define HEAT_MPI_HPP
#include <vector>
#include <mpi.h>
constexpr float PADDING_VALUE = 30.0f;

void run_heat_mpi_sync(std::vector<float>& full_grid, int H, int W, int steps);
void run_heat_mpi_async(std::vector<float>& full_grid, int H, int W, int steps);
#endif // HEAT_MPI_HPP