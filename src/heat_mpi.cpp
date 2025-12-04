#include "heat_mpi.hpp"

// Kernel 3×3
static const float K[3][3] = {
    {0.05f, 0.10f, 0.05f},
    {0.10f, 0.40f, 0.10f},
    {0.05f, 0.10f, 0.05f}
};

void run_heat_mpi_sync(std::vector<float>& full_grid, int H, int W, int steps)
{
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Partition must be divisible (simple version)
    if (H % size != 0)
    {
        if (rank == 0) std::cerr << "H must be divisible by size.\n";
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int local_H = H / size;
    int local_size = local_H * W;

    // Buffers
    std::vector<float> local_grid(local_size);
    std::vector<float> new_local(local_size);
    std::vector<float> ghost_up(W), ghost_down(W);

    // Scatter initial grid
    MPI_Scatter(full_grid.data(), local_size, MPI_FLOAT,
                local_grid.data(), local_size, MPI_FLOAT,
                0, MPI_COMM_WORLD);

    // Root no longer needs full_grid until gather; ensure buffer for gather
    if (rank != 0)
        full_grid.resize(H * W);

    const int TAG_UP   = 10;
    const int TAG_DOWN = 20;

    for (int step = 0; step < steps; ++step)
    {
        MPI_Status st;

        // --- Exchange top ghost row ---
        if (rank > 0)
        {
            MPI_Sendrecv(
                &local_grid[0], W, MPI_FLOAT,    // send local top
                rank - 1, TAG_UP,
                ghost_up.data(), W, MPI_FLOAT,   // recv ghost top
                rank - 1, TAG_DOWN,
                MPI_COMM_WORLD, &st
            );
        }
        else
        {
            std::fill(ghost_up.begin(), ghost_up.end(), PADDING_VALUE);
        }

        // --- Exchange bottom ghost row ---
        if (rank < size - 1)
        {
            MPI_Sendrecv(
                &local_grid[(local_H - 1) * W], W, MPI_FLOAT,  // send local bottom
                rank + 1, TAG_DOWN,
                ghost_down.data(), W, MPI_FLOAT,               // recv ghost bottom
                rank + 1, TAG_UP,
                MPI_COMM_WORLD, &st
            );
        }
        else
        {
            std::fill(ghost_down.begin(), ghost_down.end(), PADDING_VALUE);
        }

        // ---------------------------------------
        // Compute new values using 3×3 kernel
        // ---------------------------------------
        for (int i = 0; i < local_H; ++i)
        {
            for (int j = 0; j < W; ++j)
            {
                float acc = 0.0f;

                // Apply kernel
                for (int ki = -1; ki <= 1; ++ki)
                {
                    for (int kj = -1; kj <= 1; ++kj)
                    {
                        int ni = i + ki;
                        int nj = j + kj;
                        float val = PADDING_VALUE;

                        // --- row check (global boundary) ---
                        if (ni < 0) {
                            // use ghost_up
                            val = ghost_up[nj];
                        }
                        else if (ni >= local_H) {
                            // use ghost_down
                            val = ghost_down[nj];
                        }
                        else {
                            // interior local row
                            if (nj >= 0 && nj < W)
                                val = local_grid[ni * W + nj];
                            else
                                val = PADDING_VALUE;
                        }

                        acc += val * K[ki + 1][kj + 1];
                    }
                }

                new_local[i * W + j] = acc;
            }
        }

        // Swap
        local_grid.swap(new_local);

        MPI_Barrier(MPI_COMM_WORLD);
    }

    // Gather final matrix back to root
    MPI_Gather(local_grid.data(), local_size, MPI_FLOAT,
               full_grid.data(), local_size, MPI_FLOAT,
               0, MPI_COMM_WORLD);
}

void run_heat_mpi_async(std::vector<float>& full_grid, int H, int W, int steps)
{
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (H % size != 0) {
        if (rank == 0) std::cerr << "H must be divisible by number of processes.\n";
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int local_H = H / size;
    int local_size = local_H * W;

    // Buffers
    std::vector<float> local_grid(local_size);
    std::vector<float> new_local(local_size);
    std::vector<float> ghost_up(W), ghost_down(W);

    // Root allocates output buffer for gather
    if (rank != 0)
        full_grid.resize(H * W);

    MPI_Scatter(full_grid.data(), local_size, MPI_FLOAT,
                local_grid.data(), local_size, MPI_FLOAT,
                0, MPI_COMM_WORLD);

    const int TAG_UP   = 10;
    const int TAG_DOWN = 20;

    // We may use up to 4 requests (2 receives, 2 sends)
    MPI_Request reqs[4];
    MPI_Status stats[4];

    for (int step = 0; step < steps; ++step)
    {
        int req_count = 0;

        // --- POST NON-BLOCKING RECEIVES FIRST ---
        if (rank > 0) {
            MPI_Irecv(ghost_up.data(), W, MPI_FLOAT,
                      rank - 1, TAG_DOWN, MPI_COMM_WORLD,
                      &reqs[req_count++]);
        } else {
            std::fill(ghost_up.begin(), ghost_up.end(), PADDING_VALUE);
        }

        if (rank < size - 1) {
            MPI_Irecv(ghost_down.data(), W, MPI_FLOAT,
                      rank + 1, TAG_UP, MPI_COMM_WORLD,
                      &reqs[req_count++]);
        } else {
            std::fill(ghost_down.begin(), ghost_down.end(), PADDING_VALUE);
        }

        // --- POST NON-BLOCKING SENDS ---
        if (rank > 0) {
            MPI_Isend(&local_grid[0], W, MPI_FLOAT,
                      rank - 1, TAG_UP, MPI_COMM_WORLD,
                      &reqs[req_count++]);
        }
        if (rank < size - 1) {
            MPI_Isend(&local_grid[(local_H - 1) * W], W, MPI_FLOAT,
                      rank + 1, TAG_DOWN, MPI_COMM_WORLD,
                      &reqs[req_count++]);
        }

        // ---------------------------------------------------
        //  Compute INTERIOR rows while ghost rows are incoming
        //  Interior = rows 1 .. local_H - 2  (if >= 3 rows)
        // ---------------------------------------------------
        if (local_H >= 3)
        {
            for (int i = 1; i <= local_H - 2; i++) {
                for (int j = 0; j < W; j++) {

                    float acc = 0.0f;

                    for (int ki = -1; ki <= 1; ki++) {
                        for (int kj = -1; kj <= 1; kj++) {
                            int ni = i + ki;
                            int nj = j + kj;

                            float val = PADDING_VALUE;

                            if (nj >= 0 && nj < W &&
                                ni >= 0 && ni < local_H)
                            {
                                val = local_grid[ni * W + nj];
                            }

                            acc += val * K[ki + 1][kj + 1];
                        }
                    }

                    new_local[i * W + j] = acc;
                }
            }
        }

        // ---------------------------------------------------
        // Wait until ghost rows arrive
        // ---------------------------------------------------
        if (req_count > 0)
            MPI_Waitall(req_count, reqs, stats);

        // ---------------------------------------------------
        // Compute TOP boundary row (i = 0) using ghost_up
        // ---------------------------------------------------
        {
            int i = 0;
            for (int j = 0; j < W; j++) {

                float acc = 0.0f;

                for (int ki = -1; ki <= 1; ki++) {
                    for (int kj = -1; kj <= 1; kj++) {

                        int ni = i + ki;
                        int nj = j + kj;

                        float val = PADDING_VALUE;

                        if (nj >= 0 && nj < W) {
                            if (ni < 0)
                                val = ghost_up[nj];
                            else if (ni >= 0)
                                val = local_grid[ni * W + nj];
                        }

                        acc += val * K[ki + 1][kj + 1];
                    }
                }

                new_local[i * W + j] = acc;
            }
        }

        // ---------------------------------------------------
        // Compute BOTTOM boundary row (i = local_H - 1)
        // ---------------------------------------------------
        if (local_H >= 2)
        {
            int i = local_H - 1;

            for (int j = 0; j < W; j++) {

                float acc = 0.0f;

                for (int ki = -1; ki <= 1; ki++) {
                    for (int kj = -1; kj <= 1; kj++) {

                        int ni = i + ki;
                        int nj = j + kj;

                        float val = PADDING_VALUE;

                        if (nj >= 0 && nj < W) {
                            if (ni >= local_H)
                                val = ghost_down[nj];
                            else
                                val = local_grid[ni * W + nj];
                        }

                        acc += val * K[ki + 1][kj + 1];
                    }
                }

                new_local[i * W + j] = acc;
            }
        }

        // Swap buffers
        local_grid.swap(new_local);
    }

    // Gather global result
    MPI_Gather(local_grid.data(), local_size, MPI_FLOAT,
               full_grid.data(), local_size, MPI_FLOAT,
               0, MPI_COMM_WORLD);
}
