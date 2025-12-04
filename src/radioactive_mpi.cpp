#include "radioactive_mpi.hpp"

float calc_next_c(float C, float C_left, float C_right, float C_up, float C_down) {
    float d2x = (C_right - 2.0f * C + C_left) / (DX * DX);
    float d2y = (C_down  - 2.0f * C + C_up)   / (DY * DY);
    float diffusion = DIFF_D * (d2x + d2y);

    float adv_x = UX * (C - C_left) / DX;
    float adv_y = UY * (C - C_up)   / DY;
    float advection = adv_x + adv_y;

    float decay = (LAMBDA_DECAY + K_DEP) * C;

    float dC_dt = diffusion - advection - decay;
    float next_C = C + DT * dC_dt;

    if (next_C < 0.0f || !std::isfinite(next_C)) {
        return 0.0f;
    }
    return next_C;
}

void run_radioactive_mpi_sync(std::vector<float>& full_grid, int steps) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (H % size != 0) {
        if (rank == 0) std::cerr << "H must be divisible by number of processes\n";
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int local_H = H / size;
    int local_size = local_H * W;

    std::vector<float> local_grid(local_size);
    std::vector<float> new_local(local_size);
    std::vector<float> ghost_up(W), ghost_down(W);
    std::vector<float> buf((local_H + 2) * W);

    MPI_Scatter(full_grid.data(), local_size, MPI_FLOAT,
                local_grid.data(), local_size, MPI_FLOAT,
                0, MPI_COMM_WORLD);
    if (rank != 0) full_grid.resize(H * W);
    
    for (int t = 0; t < steps; t++) {
        MPI_Status st;

        const int TAG_UP = 0;
        const int TAG_DOWN = 1;

        // Exchange with rank - 1 (UP neighbor)
        if (rank > 0) {
            MPI_Sendrecv(
                &local_grid[0],               // send top row
                W, MPI_FLOAT,
                rank - 1, TAG_UP,

                ghost_up.data(),              // receive ghost_up
                W, MPI_FLOAT,
                rank - 1, TAG_DOWN,

                MPI_COMM_WORLD, &st
            );
        } else {
            std::fill(ghost_up.begin(), ghost_up.end(), 0.0f);
        }

        // Exchange with rank + 1 (DOWN neighbor)
        if (rank < size - 1) {
            MPI_Sendrecv(
                &local_grid[(local_H - 1) * W], // send bottom row
                W, MPI_FLOAT,
                rank + 1, TAG_DOWN,

                ghost_down.data(),              // receive ghost_down
                W, MPI_FLOAT,
                rank + 1, TAG_UP,

                MPI_COMM_WORLD, &st
            );
        } else {
            std::fill(ghost_down.begin(), ghost_down.end(), 0.0f);
        }


        std::memcpy(buf.data(), ghost_up.data(), W * sizeof(float));
        std::memcpy(buf.data() + W, local_grid.data(), local_size * sizeof(float));
        std::memcpy(buf.data() + (local_H + 1) * W, ghost_down.data(), W * sizeof(float));

        for (int i = 0; i < local_H; i++) {
            for (int j = 0; j < W; j++) {
                float C = buf[(i + 1) * W + j];
                float C_up = buf[i * W + j];
                float C_down = buf[(i + 2) * W + j];
                float C_left = (j > 0 ? buf[(i + 1) * W + j - 1] : 0.0f);
                float C_right = (j < W - 1 ? buf[(i + 1) * W + j + 1] : 0.0f);

                new_local[i * W + j] = calc_next_c(C, C_left, C_right, C_up, C_down);
            }
        }

        local_grid.swap(new_local);

        long long local_safe = 0;
        for (float v : local_grid) if (std::fabs(v) < 1e-6) local_safe++;

        long long total_safe = 0;
        MPI_Reduce(&local_safe, &total_safe, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

        MPI_Barrier(MPI_COMM_WORLD);
    }

    int stop = 1;
    MPI_Bcast(&stop, 1, MPI_INT, 0, MPI_COMM_WORLD);

    MPI_Gather(local_grid.data(), local_size, MPI_FLOAT,
               full_grid.data(), local_size, MPI_FLOAT,
               0, MPI_COMM_WORLD);
}