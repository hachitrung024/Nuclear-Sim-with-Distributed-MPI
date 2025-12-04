#include "simulation.hpp"

void run_mpi(std::vector<float>& full_grid, int steps) {
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

    for (int t = 0; t < steps; t++) {
        MPI_Status st;

        if (rank < size - 1) {
            MPI_Send(&local_grid[(local_H - 1) * W], W, MPI_FLOAT, rank + 1, 0, MPI_COMM_WORLD);
            MPI_Recv(ghost_down.data(), W, MPI_FLOAT, rank + 1, 1, MPI_COMM_WORLD, &st);
        } else {
            std::memset(ghost_down.data(), 0, W * sizeof(float));
        }

        if (rank > 0) {
            MPI_Recv(ghost_up.data(), W, MPI_FLOAT, rank - 1, 0, MPI_COMM_WORLD, &st);
            MPI_Send(&local_grid[0], W, MPI_FLOAT, rank - 1, 1, MPI_COMM_WORLD);
        } else {
            std::memset(ghost_up.data(), 0, W * sizeof(float));
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