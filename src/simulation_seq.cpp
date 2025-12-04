#include "simulation.hpp"

void run_sequential(std::vector<float>& radioactive_grid, int steps) {
    std::vector<float> next_grid = radioactive_grid;

    for (int t = 0; t < steps; ++t) {
        for (int i = 0; i < H; ++i) {
            for (int j = 0; j < W; ++j) {
                int c_idx = idx(i, j);
                float C = radioactive_grid[c_idx];

                float C_left  = (j > 0)     ? radioactive_grid[idx(i, j - 1)] : 0.0f;
                float C_right = (j < W - 1) ? radioactive_grid[idx(i, j + 1)] : 0.0f;
                float C_up    = (i > 0)     ? radioactive_grid[idx(i - 1, j)] : 0.0f;
                float C_down  = (i < H - 1) ? radioactive_grid[idx(i + 1, j)] : 0.0f;

                next_grid[c_idx] = calc_next_c(C, C_left, C_right, C_up, C_down);
            }
        }
        std::swap(radioactive_grid, next_grid);
    }
}