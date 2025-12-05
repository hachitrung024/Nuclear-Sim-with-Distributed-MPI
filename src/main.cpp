#include <stdio.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <mpi.h>
#include <map>

#include "radioactive_mpi.hpp"
#include "heat_mpi.hpp"
#include "shockwave_mpi.hpp"
#include "utils.hpp"

using namespace std;

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int world_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    char hostname[MPI_MAX_PROCESSOR_NAME];
    int name_len;
    MPI_Get_processor_name(hostname, &name_len);

    // Gather all hostnames at rank 0
    vector<char> all_hosts(world_size * MPI_MAX_PROCESSOR_NAME);

    MPI_Gather(hostname, MPI_MAX_PROCESSOR_NAME, MPI_CHAR,
            all_hosts.data(), MPI_MAX_PROCESSOR_NAME, MPI_CHAR,
            0, MPI_COMM_WORLD);

    if (world_rank == 0) {
        cout << "\n=== MPI Process Layout ===\n";

        // Count processes per host
        map<string, int> count;
        for (int r = 0; r < world_size; r++) {
            string host(&all_hosts[r * MPI_MAX_PROCESSOR_NAME]);
            count[host]++;
        }

        for (auto& kv : count) {
            cout << kv.first << " : " << kv.second << " processes\n";
        }
        cout << "==========================\n\n";
    }
    if (argc < 2) {
        if (world_rank == 0) {
            cerr << "Usage: " << argv[0] << " <mode>\n";
            cerr << "Mode: 0 = MPI synchronous, 1 = MPI asynchronous\n";
        }
        MPI_Finalize();
        return 1;
    }

    int mode = stoi(argv[1]);
    int steps = 100;

    // --- allocate ---
    vector<float> radioactive_grid;
    vector<float> heat_grid;
    vector<float> shockwave_grid(H * W, 0.0f);

    // --- rank 0 loads input ---
    if (world_rank == 0) {
        radioactive_grid = read_csv("data/radioactive_matrix.csv", H, W);
        heat_grid        = read_csv("data/heat_matrix.csv", H, W);

        long long safe_count = 0;
        for (float v : radioactive_grid)
            if (fabs(v) < 1e-6) safe_count++;

        cout << "Safe cells before radioactive simulation: "
             << safe_count << "\n";
    }

    // --- ensure other ranks allocate buffers ---
    MPI_Barrier(MPI_COMM_WORLD);
    if (world_rank != 0) {
        radioactive_grid.resize(H * W);
        heat_grid.resize(H * W);
    }

    // --- timing results ---
    double time_radio = 0;
    double time_heat = 0;
    double time_shock = 0;

    if (mode == 0) {
        if (world_rank == 0)
            cout << "Running MPI synchronous simulation...\n";

        // ===================== RADIOACTIVE =====================
        MPI_Barrier(MPI_COMM_WORLD);
        auto t1 = chrono::high_resolution_clock::now();
        run_radioactive_mpi_sync(radioactive_grid, steps);
        MPI_Barrier(MPI_COMM_WORLD);
        auto t2 = chrono::high_resolution_clock::now();
        time_radio = chrono::duration<double>(t2 - t1).count();

        // ===================== HEAT =====================
        MPI_Barrier(MPI_COMM_WORLD);
        t1 = chrono::high_resolution_clock::now();
        run_heat_mpi_sync(heat_grid, H, W, steps);
        MPI_Barrier(MPI_COMM_WORLD);
        t2 = chrono::high_resolution_clock::now();
        time_heat = chrono::duration<double>(t2 - t1).count();

        // ===================== SHOCKWAVE =====================
        MPI_Barrier(MPI_COMM_WORLD);
        t1 = chrono::high_resolution_clock::now();
        run_shockwave_mpi_sync(shockwave_grid, H, W, steps);
        MPI_Barrier(MPI_COMM_WORLD);
        t2 = chrono::high_resolution_clock::now();
        time_shock = chrono::duration<double>(t2 - t1).count();

    } else {

        if (world_rank == 0)
            cout << "Running MPI asynchronous simulation...\n";

        // ===================== RADIOACTIVE ASYNC =====================
        MPI_Barrier(MPI_COMM_WORLD);
        auto t1 = chrono::high_resolution_clock::now();
        run_radioactive_mpi_async(radioactive_grid, steps);
        MPI_Barrier(MPI_COMM_WORLD);
        auto t2 = chrono::high_resolution_clock::now();
        time_radio = chrono::duration<double>(t2 - t1).count();

        // ===================== HEAT ASYNC =====================
        MPI_Barrier(MPI_COMM_WORLD);
        t1 = chrono::high_resolution_clock::now();
        run_heat_mpi_async(heat_grid, H, W, steps);
        MPI_Barrier(MPI_COMM_WORLD);
        t2 = chrono::high_resolution_clock::now();
        time_heat = chrono::duration<double>(t2 - t1).count();

        // ===================== SHOCKWAVE ASYNC =====================
        MPI_Barrier(MPI_COMM_WORLD);
        t1 = chrono::high_resolution_clock::now();
        run_shockwave_mpi_async(shockwave_grid, H, W, steps);
        MPI_Barrier(MPI_COMM_WORLD);
        t2 = chrono::high_resolution_clock::now();
        time_shock = chrono::duration<double>(t2 - t1).count();
    }

    // ===================== OUTPUT RESULTS =====================
    if (world_rank == 0) {

        cout << "\n=== Timing Results (" << (mode == 0 ? "Sync" : "Async") << ") ===\n";
        cout << "Radioactive : " << time_radio << " sec\n";
        cout << "Heat        : " << time_heat  << " sec\n";
        cout << "Shockwave   : " << time_shock << " sec\n";

        long long safe_after = 0;
        for (float v : radioactive_grid)
            if (fabs(v) < 1e-6) safe_after++;
        cout << "Safe cells after radioactive simulation: "
             << safe_after << "\n\n";

        // Uncomment if you want to save output
        // write_csv(radioactive_grid, "output/radioactive_matrix.csv");
        // write_csv(heat_grid, "output/heat_matrix.csv");
        // write_csv(shockwave_grid, "output/shockwave_matrix.csv");
    }

    MPI_Finalize();
    return 0;
}
