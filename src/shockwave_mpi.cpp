#include "shockwave_mpi.hpp"

float compute_overpressure(double R)
{
    double W = YIELD_KG;

    if (R == 0) return 1e5; // avoid log(0)

    double Z = R * std::pow(W, -1.0/3.0);
    double U = -0.21436 + 1.35034 * std::log10(Z);

    double C[9] = {
        2.611369, -1.690128, 0.00805, 0.336743,
        -0.005162, -0.080923, -0.004785,
        0.007930, 0.000768
    };

    double logP = 0.0;
    double Ui = 1.0;

    for (int i = 0; i < 9; i++) {
        logP += C[i] * Ui;
        Ui *= U;
    }

    return (float)std::pow(10.0, logP);
}

void run_shockwave_mpi_sync(std::vector<float>& full_grid,
                            int H, int W,
                            int steps)
{
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (H % size != 0)
    {
        if (rank == 0)
            std::cerr << "H must be divisible by number of processes.\n";
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    const int cx = H / 2;
    const int cy = W / 2;

    int local_H = H / size;
    int local_size = local_H * W;

    // Allocate local grids
    std::vector<float> local_grid(local_size, 0.0f);
    std::vector<float> ghost_up(W), ghost_down(W);

    // Root ensures full_grid is allocated for gather
    if (rank != 0)
        full_grid.resize(H * W);

    // Scatter initial grid
    MPI_Scatter(full_grid.data(), local_size, MPI_FLOAT,
                local_grid.data(), local_size, MPI_FLOAT,
                0, MPI_COMM_WORLD);

    const int TAG_UP   = 100;
    const int TAG_DOWN = 200;

    // Global x-coordinate range for this rank:
    int global_row_start = rank * local_H;
    int global_row_end   = global_row_start + local_H - 1;

    //-------------------------------------------------
    // Main simulation loop
    //-------------------------------------------------
    for (int t = 0; t <= steps; t++)
    {
        float Rmax     = SOUND_SPEED * t;
        float Rmax_cell = Rmax / CELL_SIZE;

        // Local bounding box of wave that might affect this rank
        int imin_global = std::max(0, cx - (int)Rmax_cell);
        int imax_global = std::min(H - 1, cx + (int)Rmax_cell);

        // Convert to local index
        int imin = std::max(0, imin_global - global_row_start);
        int imax = std::min(local_H - 1, imax_global - global_row_start);

        // --- Exchange ghost rows so the wave can propagate across boundaries ---
        MPI_Status st;

        // Top ghost
        if (rank > 0) {
            MPI_Sendrecv(
                &local_grid[0], W, MPI_FLOAT,
                rank - 1, TAG_UP,

                ghost_up.data(), W, MPI_FLOAT,
                rank - 1, TAG_DOWN,
                MPI_COMM_WORLD, &st
            );
        } else {
            std::fill(ghost_up.begin(), ghost_up.end(), 0.0f);
        }

        // Bottom ghost
        if (rank < size - 1) {
            MPI_Sendrecv(
                &local_grid[(local_H - 1) * W], W, MPI_FLOAT,
                rank + 1, TAG_DOWN,

                ghost_down.data(), W, MPI_FLOAT,
                rank + 1, TAG_UP,
                MPI_COMM_WORLD, &st
            );
        } else {
            std::fill(ghost_down.begin(), ghost_down.end(), 0.0f);
        }

        //-------------------------------------------------
        // Compute wave updates inside this local region
        //-------------------------------------------------

        for (int li = imin; li <= imax; li++)
        {
            int gi = global_row_start + li;   // global row index
            float dx_cell = (float)(gi - cx);

            float remain = Rmax_cell * Rmax_cell - dx_cell * dx_cell;
            if (remain < 0) continue;

            float dy_max_cell = std::sqrt(remain);

            int jmin = std::max(0,  (int)(cy - dy_max_cell));
            int jmax = std::min(W-1,(int)(cy + dy_max_cell));

            for (int j = jmin; j <= jmax; j++)
            {
                float &cell = local_grid[li * W + j];
                if (cell > 0.0f)
                    continue;

                float dx = (gi - cx) * CELL_SIZE;
                float dy = (j  - cy) * CELL_SIZE;
                float R  = std::sqrt(dx * dx + dy * dy);

                float arrival_time = R / SOUND_SPEED;

                if (arrival_time <= t)
                    cell = compute_overpressure(R);
            }
        }
    }

    //-------------------------------------------------
    // Gather results back to root
    //-------------------------------------------------
    MPI_Gather(local_grid.data(), local_size, MPI_FLOAT,
               full_grid.data(), local_size, MPI_FLOAT,
               0, MPI_COMM_WORLD);
}

void run_shockwave_mpi_async(std::vector<float>& full_grid,
                             int H, int W,
                             int steps)
{
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (H % size != 0) {
        if (rank == 0) 
            std::cerr << "H must be divisible by number of processes.\n";
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    const int cx = H / 2;
    const int cy = W / 2;

    int local_H = H / size;
    int local_size = local_H * W;

    // Local memories
    std::vector<float> local_grid(local_size, 0.0f);
    std::vector<float> ghost_up(W), ghost_down(W);

    // Root ensures buffer for gather
    if (rank != 0)
        full_grid.resize(H * W);

    // Scatter initial grid
    MPI_Scatter(full_grid.data(), local_size, MPI_FLOAT,
                local_grid.data(), local_size, MPI_FLOAT,
                0, MPI_COMM_WORLD);

    const int TAG_UP   = 100;
    const int TAG_DOWN = 200;

    // Global coordinate for this block
    int global_row_start = rank * local_H;
    int global_row_end   = global_row_start + local_H - 1;

    // Requests (max 4: 2 recv, 2 send)
    MPI_Request reqs[4];
    MPI_Status stats[4];

    //------------------------------------
    // Main simulation
    //------------------------------------
    for (int t = 0; t <= steps; t++)
    {
        float Rmax      = SOUND_SPEED * (float)t;
        float Rmax_cell = Rmax / CELL_SIZE;

        // bounding rows (global)
        int imin_g = std::max(0, cx - (int)Rmax_cell);
        int imax_g = std::min(H - 1, cx + (int)Rmax_cell);

        // convert to local
        int imin = std::max(0, imin_g - global_row_start);
        int imax = std::min(local_H - 1, imax_g - global_row_start);

        //-----------------------------------------------------
        // 1) ASYNC COMMUNICATION (Irecv first, then Isend)
        //-----------------------------------------------------
        int req_count = 0;

        // Upper neighbor
        if (rank > 0) {
            MPI_Irecv(ghost_up.data(), W, MPI_FLOAT,
                      rank - 1, TAG_DOWN, MPI_COMM_WORLD,
                      &reqs[req_count++]);
        } else {
            std::fill(ghost_up.begin(), ghost_up.end(), 0.0f);
        }

        // Lower neighbor
        if (rank < size - 1) {
            MPI_Irecv(ghost_down.data(), W, MPI_FLOAT,
                      rank + 1, TAG_UP, MPI_COMM_WORLD,
                      &reqs[req_count++]);
        } else {
            std::fill(ghost_down.begin(), ghost_down.end(), 0.0f);
        }

        // Send top row
        if (rank > 0) {
            MPI_Isend(&local_grid[0], W, MPI_FLOAT,
                      rank - 1, TAG_UP, MPI_COMM_WORLD,
                      &reqs[req_count++]);
        }

        // Send bottom row
        if (rank < size - 1) {
            MPI_Isend(&local_grid[(local_H - 1) * W], W, MPI_FLOAT,
                      rank + 1, TAG_DOWN, MPI_COMM_WORLD,
                      &reqs[req_count++]);
        }

        //-----------------------------------------------------
        // 2) COMPUTE INTERIOR CELLS (independent of ghost)
        //-----------------------------------------------------
        // interior rows = [1 .. local_H-2]
        if (local_H >= 3)
        {
            int i_start = std::max(1, imin);
            int i_end   = std::min(local_H - 2, imax);

            for (int li = i_start; li <= i_end; li++)
            {
                int gi = global_row_start + li;
                float dx_cell = (float)(gi - cx);

                float remain = Rmax_cell * Rmax_cell - dx_cell * dx_cell;
                if (remain < 0) continue;

                float dy_max_cell = std::sqrt(remain);

                int jmin = std::max(0, (int)(cy - dy_max_cell));
                int jmax = std::min(W-1,(int)(cy + dy_max_cell));

                for (int j = jmin; j <= jmax; j++)
                {
                    float &cell = local_grid[li * W + j];
                    if (cell > 0.0f) continue;

                    double dx = (double)(gi - cx) * CELL_SIZE;
                    double dy = (double)(j  - cy) * CELL_SIZE;
                    double R  = std::sqrt(dx*dx + dy*dy);
                    double arrival = R / SOUND_SPEED;

                    if (arrival <= t)
                        cell = compute_overpressure((float)R);
                }
            }
        }

        //-----------------------------------------------------
        // 3) WAIT FOR COMMUNICATION
        //-----------------------------------------------------
        if (req_count > 0)
            MPI_Waitall(req_count, reqs, stats);

        //-----------------------------------------------------
        // 4) COMPUTE BOUNDARY ROWS (need ghosts)
        //-----------------------------------------------------

        // row 0 (if it intersects bounding region)
        if (imin <= 0 && 0 <= imax)
        {
            int li = 0;
            int gi = global_row_start + li;

            float dx_cell = (float)(gi - cx);
            float remain  = Rmax_cell * Rmax_cell - dx_cell * dx_cell;
            if (remain >= 0)
            {
                float dy_max_cell = std::sqrt(remain);

                int jmin = std::max(0, (int)(cy - dy_max_cell));
                int jmax = std::min(W-1,(int)(cy + dy_max_cell));

                for (int j = jmin; j <= jmax; j++)
                {
                    float &cell = local_grid[li * W + j];
                    if (cell > 0.0f) continue;

                    double dx = (double)(gi - cx) * CELL_SIZE;
                    double dy = (double)(j  - cy) * CELL_SIZE;
                    double R  = std::sqrt(dx*dx + dy*dy);
                    double arrival = R / SOUND_SPEED;

                    if (arrival <= t)
                        cell = compute_overpressure((float)R);
                }
            }
        }

        // row local_H - 1
        if (imin <= local_H - 1 && local_H - 1 <= imax)
        {
            int li = local_H - 1;
            int gi = global_row_start + li;

            float dx_cell = (float)(gi - cx);
            float remain  = Rmax_cell * Rmax_cell - dx_cell * dx_cell;
            if (remain >= 0)
            {
                float dy_max_cell = std::sqrt(remain);

                int jmin = std::max(0, (int)(cy - dy_max_cell));
                int jmax = std::min(W-1,(int)(cy + dy_max_cell));

                for (int j = jmin; j <= jmax; j++)
                {
                    float &cell = local_grid[li * W + j];
                    if (cell > 0.0f) continue;

                    double dx = (double)(gi - cx) * CELL_SIZE;
                    double dy = (double)(j  - cy) * CELL_SIZE;
                    double R  = std::sqrt(dx*dx + dy*dy);
                    double arrival = R / SOUND_SPEED;

                    if (arrival <= t)
                        cell = compute_overpressure((float)R);
                }
            }
        }

        // done step
    }

    //-----------------------------------------------------
    // 5) GATHER final results
    //-----------------------------------------------------
    MPI_Gather(local_grid.data(), local_size, MPI_FLOAT,
               full_grid.data(), local_size, MPI_FLOAT,
               0, MPI_COMM_WORLD);
}
