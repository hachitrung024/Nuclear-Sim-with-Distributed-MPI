#include "utils.hpp"

// Function to read a CSV file into a vector
std::vector<float> read_csv(const std::string& path, int height, int width) {
    std::vector<float> grid(height * width);
    std::ifstream fin(path);

    if (!fin.is_open()) {
        throw std::runtime_error("Cannot open CSV file: " + path);
    }

    std::string line;
    int row = 0;

    // Read each line of the CSV
    while (getline(fin, line) && row < height) {
        std::stringstream ss(line);
        std::string cell;
        int col = 0;

        // Read each cell in the line, split by commas
        while (getline(ss, cell, ',') && col < width) {
            grid[idx(row, col)] = std::stof(cell);  // Sử dụng hàm idx
            col++;
        }

        row++;
    }

    fin.close();
    return grid;
}

void write_csv(const std::vector<float>& data, const std::string& path) {
    std::ofstream fout(path);

    if (!fout.is_open()) {
        throw std::runtime_error("Cannot open output file: " + path);
    }

    // Set output formatting to fixed-point notation with 2 decimal places
    fout << std::fixed << std::setprecision(2);

    // Write each row in the matrix
    for (int i = 0; i < H; i++) {
        for (int j = 0; j < W; j++) {
            fout << data[idx(i, j)];
            if (j < W - 1) fout << ",";
        }
        fout << "\n";
    }

    fout.close();
}
