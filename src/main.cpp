#include <stdio.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <mpi.h>
#include "radioactive_mpi.hpp"
#include "utils.hpp"

using namespace std;

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int world_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    if(argc < 2) {
        if (world_rank == 0) {
            cerr << "Usage: " << argv[0] << " <mode>\n";
            cerr << "Mode: 0 for sequential, 1 for MPI\n";
        }
        MPI_Finalize();
        return 1;
    }

    int mode = stoi(argv[1]);
    int steps = 100;

    vector<float> radioactive_grid;

    if (world_rank == 0) {
        radioactive_grid = read_csv("data/radioactive_matrix.csv", H, W);        
        long long safe_count = 0;
        for (float v : radioactive_grid) if (fabs(v) < 1e-6) safe_count++;
        cout << "Safe cells before simulation: " << safe_count << "\n";
    }

    MPI_Barrier(MPI_COMM_WORLD);
    auto start = chrono::high_resolution_clock::now();

    if (mode == 0) {
        if (world_rank == 0) {
            cout << "Running sequential simulation...\n";
            // run_sequential(radioactive_grid, steps);
        }
    } else if (mode == 1) {
        if (world_rank == 0) {
            cout << "Running MPI simulation...\n";
        }

        if (world_rank != 0) {
            radioactive_grid.resize(H * W);
        }

        run_radioactive_mpi_sync(radioactive_grid, steps);
    }


    MPI_Barrier(MPI_COMM_WORLD);
    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double> elapsed = end - start;

    if (world_rank == 0) {
        cout << "Elapsed time: " << elapsed.count() << " seconds\n";
        
        long long safe_after = 0;
        for (float v : radioactive_grid) if (fabs(v) < 1e-6) safe_after++;
        cout << "Safe cells after simulation: " << safe_after << "\n";

        // write_csv(radioactive_grid, "output/radioactive_matrix.csv");
    }

    MPI_Finalize();
    return 0;
}