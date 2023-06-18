#include <iostream>
#include <fstream>
#include <complex>
#include <vector>
#include "edmd_functions.hpp"
#include <random>
#include <utility>

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



std::vector<std::pair<int, int>> generateSamples(int num, int numsamplepairs) {
    std::random_device rd;  // used to obtain a seed for the random number engine
    std::mt19937 generator(rd()); // Standard mersenne_twister_engine seeded with rd()
    std::uniform_int_distribution<> distrib(0, num - 1);

    std::vector<std::pair<int, int>> samples;
    samples.reserve(numsamplepairs);
    for (int i = 0; i < numsamplepairs; ++i) {
        int a = distrib(generator);
        int b = distrib(generator);

        while(a == b) {
            b = distrib(generator);  // Regenerate 'b' if it's the same as 'a'
        }
        samples.push_back(std::make_pair(a, b));
    }
    return samples;
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

    // Call the function
    std::vector<std::vector<std::complex<double>>> psisAll = readPsis(filename, num);
    std::complex<double> mean(0.0,0.0);
    for(int i = 0;i<psisAll[0].size();i++) mean+=psisAll[psisAll.size()-1][i];
    mean/=num;


    double binSize = 2.0;
    int numBins = static_cast<int>(0.5 * L * std::sqrt(2.0) / binSize);
    std::vector<double> binEdges(numBins + 1);
    std::vector<double> binCenters(numBins);
    for (int i = 0; i <= numBins; ++i) {
        binEdges[i] = i * binSize;
    }
    for (int i = 0; i < numBins; ++i) {
        binCenters[i] = (binEdges[i] + binEdges[i + 1]) / 2.0;
    }
    std::vector<int> ocfcounts(binCenters.size(),0);
    std::vector<double> ocf(binCenters.size(),0.0);

    int numsamplepairs = 10*num;

    std::vector<int> frames;
    for(int frame = 0;frame<34000;frame+=500) frames.push_back(frame);

    for(int frameind = 40; frameind<frames.size();frameind++)
    {
        std::cout << frameind << std::endl;
        int frame = frames[frameind];

        std::vector<double> data = readPositionFrame(simid, frame);

        std::vector<std::vector<double>> positions(data.size()/2, std::vector<double>(2));
        for (int i = 0; i < data.size(); i++) {
            positions[i/2][i%2] = data[i];
        }

        
        
        std::complex<double> meansqpsi(0.0,0.0);
        for(int i = 0;i<psisAll[frameind].size();i++) meansqpsi+=std::abs(psisAll[frameind][i])*std::abs(psisAll[frameind][i]);
        meansqpsi/=num;


        std::vector<std::pair<int, int>> samples = generateSamples(num, numsamplepairs);
        for(int i = 0; i< samples.size();i++) 
        {
            int ind1 = samples[i].first;
            int ind2 = samples[i].second;
            std::vector<double> r1 = positions[ind1];
            std::vector<double> r2 = positions[ind2];
            std::vector<double>r12 = {r2[0]-r1[0],r2[1]-r1[1]};
            PBCvec(r12,L);
            double dist = std::sqrt(r12[0]*r12[0] + r12[1]*r12[1]);
            
            for(int j = 0; j < binEdges.size() - 1; j++) 
            {
                if(binEdges[j] <= dist && dist < binEdges[j+1]) 
                {
                    ocf[j] += std::real(std::conj(psisAll[frameind][ind1])*psisAll[frameind][ind2]/meansqpsi);
                    ocfcounts[j]++;
                    break;
                }
            }
        }
    }
    for(int i = 0;i<ocf.size();i++) if (ocfcounts[i]>0) ocf[i]/=ocfcounts[i];

    for(int i = 0;i<ocf.size();i++) std::cout << binCenters[i] << "\t"<<ocf[i] << std::endl;
    


    



    // Just for debugging, print out the size of complexData
    // std::cout <<  mean << std::endl;

    return 0;
}


// std::cout << samples[i].first << ", "<<samples[i].second <<std::endl;