// This code computes the orientational order parameter given the positions.
// It also computes the coarse-grained density.

#include <iostream>
#include <vector>
#include <unordered_map>
#include <cmath>
#include <fstream>
#include "edmd_functions.hpp"
#include <cstdio>
#include <complex>

void writeMatrixToBinary(const std::string& filename, const std::vector<std::vector<double>>& matrix) {
    std::ofstream outfile(filename, std::ios::binary | std::ios::app);
    for (const auto& row : matrix) {
        outfile.write(reinterpret_cast<const char*>(&row[0]), row.size() * sizeof(double));
    }
    outfile.close();
}

void writeCellCentersToBinary(const std::string& filename, const std::vector<double>& vec) {
    std::ofstream outfile(filename, std::ios::binary | std::ios::app);
    outfile.write(reinterpret_cast<const char*>(&vec[0]), vec.size() * sizeof(double));
    outfile.close();
}

void writeNumberDens(double cellsize0, int simid, double L, long long num, double pi, double R) {
    int ncells = static_cast<int>(std::floor(L / (cellsize0)));
    double cellsize = L / ncells;
    std::vector<double> cellcenters(ncells);
    for (int i = 0; i < ncells; ++i) {
        cellcenters[i] = -0.5 * L + 0.5 * cellsize + cellsize * i;
    }

    writeCellCentersToBinary("cellcenters.bin", cellcenters);

    std::ofstream outfile("ncells.bin", std::ios::binary | std::ios::app);
    outfile.write(reinterpret_cast<const char*>(&ncells), sizeof(int));
    outfile.close();

    for(int frame = 0; frame < 40000; frame+=500) {
        std::cout << frame << std::endl;
        std::vector<double> data = readPositionFrame(simid, frame);

        std::vector<std::vector<double>> positions(data.size()/2, std::vector<double>(2));
        for (int i = 0; i < data.size(); i++) {
            positions[i/2][i%2] = data[i];
        }

        std::vector<std::vector<int>> particle2cell = AssignCells(cellcenters, positions, num);
        std::unordered_map<std::pair<int, int>, std::vector<int>> cell2particle = GenCell2Particle(particle2cell, num, ncells);

        std::vector<std::vector<double>> cellDensity(ncells, std::vector<double>(ncells, 0.0));

        for (auto& item : cell2particle) {
            std::pair<int, int> cell = item.first;
            std::vector<int> particles = item.second;
            cellDensity[cell.first][cell.second] = pi * particles.size() / (cellsize * cellsize);
        }

        writeMatrixToBinary("density.bin", cellDensity);
    }
}

std::vector<std::complex<double>> computePsis(int frame, int simid, double L, double R, int num)
{
    
    int ncells = static_cast<int>(std::floor(L / (2*R)));
    double cellsize = L / ncells;
    std::vector<double> cellcenters(ncells);
    for (int i = 0; i < ncells; ++i) {
        cellcenters[i] = -0.5 * L + 0.5 * cellsize + cellsize * i;
    }
    
    std::vector<double> data = readPositionFrame(simid, frame);

    std::vector<std::vector<double>> positions(data.size()/2, std::vector<double>(2));
    for (int i = 0; i < data.size(); i++) {
        positions[i/2][i%2] = data[i];
    }

    std::vector<std::vector<int>> particle2cell = AssignCells(cellcenters, positions, num);
    std::unordered_map<std::pair<int, int>, std::vector<int>> cell2particle = GenCell2Particle(particle2cell, num, ncells);

    std::vector<std::complex<double>> psis(num, std::complex<double>(0, 0));


    for(int p = 0;p<num;p++)
    {
        std::vector<int> cellind = particle2cell[p];
        int c0 = cellind[0]; int c1 = cellind[1];
        int c0p1 = cmod((c0+1),ncells);
        int c0m1 = cmod((c0-1),ncells);
        int c1p1 = cmod((c1+1),ncells);
        int c1m1 = cmod((c1-1),ncells);
        std::vector<std::pair<int, int>> neighinds = { {c0, c1},{c0, c1m1},{c0, c1p1},{c0m1, c1},{c0m1, c1m1},{c0m1, c1p1},{c0p1, c1},{c0p1, c1m1},{c0p1, c1p1}  };

        std::vector<int> neighbors;
        for (auto i = 0; i < neighinds.size(); i++) 
        {
            auto iter = cell2particle.find(neighinds[i]);
            if(iter != cell2particle.end())
            {
                neighbors.insert(neighbors.end(), iter->second.begin(), iter->second.end());
            }
        }
        neighbors.erase(std::remove(neighbors.begin(), neighbors.end(), p), neighbors.end());

        std::vector<std::pair<double, std::vector<double>>> distanceAndRij;
        for (int i = 0; i < neighbors.size(); i++) 
        {
            std::vector<double> rij = {positions[p][0]-positions[neighbors[i]][0],positions[p][1]-positions[neighbors[i]][1]};
            PBCvec(rij,L);
            double dist = std::sqrt(rij[0]*rij[0] + rij[1]*rij[1]);
            distanceAndRij.push_back({dist, rij});
        }

        // Sort by distance
        std::sort(distanceAndRij.begin(), distanceAndRij.end());

        

        // Vector to store the rij vectors corresponding to the smallest distances
        std::vector<std::vector<double>> closestRij;
        int limit = distanceAndRij.size() < 6 ? distanceAndRij.size() : 6;
        for(int i = 0; i < limit; i++)
        {
            closestRij.push_back(distanceAndRij[i].second);
        }
        
        for(int i = 0;i<closestRij.size();i++) psis[p]+=std::exp(std::complex<double>(0, 1) * 6.0 * std::atan2(closestRij[i][1], closestRij[i][0]));
        psis[p]/=limit;
        
    }

    return psis;
}

#include <fstream>

void writePsi(const std::string& filename, const std::vector<std::complex<double>>& vec) 
{
    std::ofstream out(filename, std::ios::binary | std::ios::app);
    if (!out) 
    {
        std::cerr << "Cannot open file: " << filename << std::endl;
        return;
    }
    
    for (const auto& val : vec) 
    {
        double real = val.real();
        double imag = val.imag();
        out.write(reinterpret_cast<char*>(&real), sizeof(double));
        out.write(reinterpret_cast<char*>(&imag), sizeof(double));
    }
    out.close();
}



int main(int argc, char* argv[]) {


    if (argc != 3) {  // If the number of arguments is not 2 (program name + integer)
        std::cerr << "Usage: " << argv[0] << " simid L \n";
        return 1;
    }

    int simid = std::atoi(argv[1]);  // Convert the second argument to an integer

    double L = std::atof(argv[2]);  // atof converts C-string to double

    std::remove("density.bin");
    std::remove("cellcenters.bin");
    std::remove("ncells.bin");
    std::remove("psis.bin");


    

    double R = 1;
    long long num = 256*256;
    double pi = 3.14159265358979;
    double phi = num*pi/(L*L);



    double cellsize0= 32;

    // writeNumberDens(cellsize0, simid, L, num, pi, R);


    for(int frame = 0;frame<34000;frame+=500)
    {
        
        std::cout << frame <<std::endl;
        
        std::vector<std::complex<double>> psis = computePsis(frame, simid,  L,  R,  num);
        writePsi("psis.bin", psis); 

        std::complex<double> meanPsi(0,0);
        for(int p = 0;p<num;p++) meanPsi+=psis[p];
        meanPsi/=num;
        std::cout << meanPsi <<std::endl; 
    }




        
    }
    
    // for(double i =0.;i<2*pi;i+=pi/100) std::cout << std::atan2(std::sin(i),std::cos(i))<<std::endl;

       // if (p==0)
        // {
        //     std::vector<std::vector<double>> closestRij = {{1/std::sqrt(2), 1/std::sqrt(2)},{-0.258819, 0.965926},{-0.965926, 0.258819},{-1/std::sqrt(2), -1/std::sqrt(2)},{0.258819, -0.965926},{0.965926, -0.258819}};//{{1, 0}, {0.5, std::sqrt(3)/2}, {-0.5, std::sqrt(3)/2}, {-1, 0}, {-0.5, -std::sqrt(3)/2}, {0.5, -std::sqrt(3)/2}};
        //     for(int i = 0;i<closestRij.size();i++) psis[p]+=std::exp(std::complex<double>(0, 1) * 6.0 * std::atan2(closestRij[i][1], closestRij[i][0]));
        //     psis[p]/=limit;
        //     std::cout << psis[p]<<std::endl;
        // }
        // for(int i = 0;i<distanceAndRij.size();i++) std::cout << std::get<0>(distanceAndRij[i]) <<std::endl;

        // std::cout << psis[p]<<std::endl;

    


