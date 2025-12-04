#ifndef SHOCKWAVE_MPI_HPP
#define SHOCKWAVE_MPI_HPP

#include <vector>
#include <string>
#include <cmath>
#include <iostream>
#include <cmath>
#include <mpi.h>

inline const int GRID_SIZE = 4000;
inline const double CELL_SIZE = 10.0;     // meters per cell
inline const double SOUND_SPEED = 343.0;  // m/s
inline const double YIELD_KT = 5000.0;    // kilotons
inline const double YIELD_KG = YIELD_KT * 1000000.0; // TNT kg
inline const int CENTER = GRID_SIZE / 2;
inline const int SIM_TIME = 100;          // seconds

float compute_overpressure(double R);   // Pso at distance R (meters)
void run_shockwave_mpi_sync(std::vector<float>& full_grid,
                            int H, int W,
                            int steps);
void run_shockwave_mpi_async(std::vector<float>& full_grid,
                             int H, int W,
                             int steps);

#endif // SHOCKWAVE_MPI_HPP