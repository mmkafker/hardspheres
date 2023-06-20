#include <iostream>
#include <fstream>
#include <complex>
#include <vector>
#include "edmd_functions.hpp"
#include <random>
#include <utility>
#include <cmath>

std::vector<std::vector<std::complex<double>>> readPsis(const std::string& filename, int num) 
{
    std::ifstream in(filename, std::ios::binary);
    if (!in) 
    {
        std::cerr << "Cannot open file: " << filename << std::endl;
        return {};
    }

    in.seekg(0, std::ios::end);
    std::streamsize size = in.tellg();
    in.seekg(0, std::ios::beg);

    int num_complex_numbers = size / (2 * sizeof(double)); // two doubles make one complex number
    int num_rows = num_complex_numbers / num;

    std::vector<std::vector<std::complex<double>>> data(num_rows, std::vector<std::complex<double>>(num));
    for(int i = 0; i < num_rows; i++)
    {
        for(int j = 0; j < num; j++)
        {
            double real = 0.0;
            double imag = 0.0;
            in.read(reinterpret_cast<char*>(&real), sizeof(double));
            in.read(reinterpret_cast<char*>(&imag), sizeof(double));
            data[i][j] = std::complex<double>(real, imag);
        }
    }

    in.close();

    return data;
}


void writeMatrixToBinary(const std::string& filename, const std::vector<std::vector<double>>& matrix) {
    std::ofstream outfile(filename, std::ios::binary | std::ios::app);
    for (const auto& row : matrix) {
        outfile.write(reinterpret_cast<const char*>(&row[0]), row.size() * sizeof(double));
    }
    outfile.close();
}


    


int main(int argc, char* argv[]) {

    if (argc != 3) {  // If the number of arguments is not 2 (program name + integer)
        std::cerr << "Usage: " << argv[0] << " simid L \n";
        return 1;
    }

    int simid = std::atoi(argv[1]);  // Convert the second argument to an integer
    double L = std::atof(argv[2]);  // atof converts C-string to double

    const std::string filename = "psis.bin"; // replace with your file name
    const int num = 256*256; // replace with your number

    double cellsize0 = 16;

    double pi = std::acos(-1);

    // Call the function
    std::vector<std::vector<std::complex<double>>> psisAll = readPsis(filename, num);


    int ncells = static_cast<int>(std::floor(L / (cellsize0)));
    double cellsize = L / ncells;
    std::vector<double> cellcenters(ncells);
    for (int i = 0; i < ncells; ++i) cellcenters[i] = -0.5 * L + 0.5 * cellsize + cellsize * i;

    std::ofstream outfile("ncells.bin", std::ios::binary);
    outfile.write(reinterpret_cast<const char*>(&ncells), sizeof(int));
    outfile.close();

    int frameind = -1;

    for(int frame = 0; frame < 40000; frame+=500) {

        frameind++;
        std::complex<double> mean(0.0,0.0);
        for(int i = 0;i<num;i++) mean+=psisAll[frameind][i];
        mean/=num;

        std::cout << frame << std::endl;
        std::vector<double> data = readPositionFrame(simid, frame);

        std::vector<std::vector<double>> positions(data.size()/2, std::vector<double>(2));
        for (int i = 0; i < data.size(); i++) {
            positions[i/2][i%2] = data[i];
        }



        std::vector<std::vector<int>> particle2cell = AssignCells(cellcenters, positions, num);
        std::unordered_map<std::pair<int, int>, std::vector<int>> cell2particle = GenCell2Particle(particle2cell, num, ncells);

        std::vector<std::vector<double>> cellDensity(ncells, std::vector<double>(ncells, 0.0));

        std::vector<std::vector<double>> cellProjection(ncells, std::vector<double>(ncells, 0.0));

        for (auto& item : cell2particle) {
            std::pair<int, int> cell = item.first;
            std::vector<int> particles = item.second;
            cellDensity[cell.first][cell.second] = pi * particles.size() / (cellsize * cellsize);

            std::complex<double> meanpsij(0.0,0.0);
            for(int i = 0;i<particles.size();i++)
            {
                meanpsij += psisAll[frameind][particles[i]];
            }
            meanpsij/=particles.size();
            double dotProduct = (mean.real() * meanpsij.real()) + (mean.imag() * meanpsij.imag());
            double magnitudeProduct = std::abs(mean) * std::abs(meanpsij);
            cellProjection[cell.first][cell.second] = dotProduct/magnitudeProduct;

        }

        writeMatrixToBinary("density.bin", cellDensity);
        writeMatrixToBinary("projection.bin", cellProjection);
    }





    return 0;

}