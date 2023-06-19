// Compute static structure factor, extract first Bragg peak, then use it to compute positional correlation function.
#include <iostream>
#include <fstream>
#include <complex>
#include <vector>
#include "edmd_functions.hpp"
#include <random>
#include <utility>
#include <cuda_runtime.h>
#include <cuComplex.h>

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



// CUDA Kernel
__global__ void computeS(cuDoubleComplex* d_S, const double* d_xij, const double* d_yij, const double* d_qx, const double* d_qy, const int* d_numseps, const int* d_numq,const int* d_num) {
    int iq = threadIdx.x + blockIdx.x * blockDim.x;

    // Check if index is within range
    if(iq >= (*d_numq) * (*d_numq)) return;

    // Calculate qx and qy based on thread index
    double qx = d_qx[iq/(*d_numq)];
    double qy = d_qy[iq%(*d_numq)];

    // Initialize sum as a complex number with zero real and imaginary parts
    cuDoubleComplex sum = make_cuDoubleComplex(0.0, 0.0);

    for (int i = 0; i < *d_numseps; i++) {

        if(i%1000000 ==0) printf("iq = %d, i = %d of %d\n",iq,i,*d_numseps);
        // Compute e^(i(qx*xij + qy*yij))
        double exponent = qx * d_xij[i] + qy * d_yij[i];
        cuDoubleComplex value = make_cuDoubleComplex(cos(exponent), sin(exponent));

        // Add value to sum
        sum = cuCadd(sum, value);
    }

    // Scale the sum by 1/numseps
    sum = make_cuDoubleComplex(1+cuCreal(sum) / (*d_num), cuCimag(sum) / (*d_num));

    // Store the result in the d_S array
    d_S[iq] = sum;
}

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


int main(int argc, char* argv[]) {

    if (argc != 3) {  // If the number of arguments is not 2 (program name + integer)
        std::cerr << "Usage: " << argv[0] << " simid L \n";
        return 1;
    }

    int simid = std::atoi(argv[1]);  // Convert the second argument to an integer
    double L = std::atof(argv[2]);  // atof converts C-string to double

    const int num = 256*256; // replace with your number
    std::vector<std::vector<std::complex<double>>> psisAll = readPsis("psis.bin", num);
    

    // Compute displacements for the static structure factor
    //////////////////////////////////////////////////
    int numsamplepairs = 10*num;

    std::vector<int> frames;
    for(int frame = 0;frame<40000;frame+=500) frames.push_back(frame);

    std::vector<std::vector<double>> allSeps;

    for(int frameind = 40; frameind<frames.size();frameind++)
    {
        // std::cout << frameind << std::endl;
        int frame = frames[frameind];

        std::vector<double> data = readPositionFrame(simid, frame);

        std::vector<std::vector<double>> positions(data.size()/2, std::vector<double>(2));
        for (int i = 0; i < data.size(); i++) {
            positions[i/2][i%2] = data[i];
        }

        std::complex<double> meanpsi(0.0,0.0);
        for(int i = 0;i<num;i++) meanpsi+=psisAll[frameind][i];
        meanpsi/=num;
        double theta = std::atan2(std::imag(meanpsi),std::real(meanpsi));


        std::vector<std::pair<int, int>> samples = generateSamples(num, numsamplepairs);

        std::vector<std::vector<double>> sepsFrame;

        for(int i = 0; i< samples.size();i++) 
        {
            int ind1 = samples[i].first;
            int ind2 = samples[i].second;
            std::vector<double> r1 = positions[ind1];
            std::vector<double> r2 = positions[ind2];
            std::vector<double> r1r = {r1[0]*std::cos(theta)+r1[1]*std::sin(theta), -r1[0]*std::sin(theta)+r1[1]*std::cos(theta)};
            std::vector<double> r2r = {r2[0]*std::cos(theta)+r2[1]*std::sin(theta), -r2[0]*std::sin(theta)+r2[1]*std::cos(theta)};
            std::vector<double> r12 = {r2r[0]-r1r[0],r2r[1]-r1r[1]};
            PBCvec(r12,L);
            sepsFrame.push_back(r12);
        }
        allSeps.insert(allSeps.end(), sepsFrame.begin(), sepsFrame.end());
    }
    //////////////////////////////////////////////////


    // Compute static structure factor on the GPU
    //////////////////////////////////////////////////
    std::vector<double> xij;
    std::vector<double> yij;

    for (const auto& sep : allSeps) {
        xij.push_back(sep[0]);
        yij.push_back(sep[1]);
    }

    std::vector<double> qx, qy;
    for (double i = -10; i <= 10; i += 0.5) {
        qx.push_back(i);
        qy.push_back(i);
    }
    int numseps = xij.size();
    int numq = qx.size();

    double* d_xij;
    double* d_yij;
    double* d_qx;
    double* d_qy;
    int* d_numseps;
    int* d_numq;
    int* d_num;
    cuDoubleComplex* d_S;
    

    int size = numseps * sizeof(double);

    // allocate GPU memory
    cudaMalloc((void**)&d_xij, size);
    cudaMalloc((void**)&d_yij, size);
    cudaMalloc((void**)&d_qx, qx.size() * sizeof(double));
    cudaMalloc((void**)&d_qy, qy.size() * sizeof(double));
    cudaMalloc((void**)&d_numseps, sizeof(int));
    cudaMalloc((void**)&d_numq, sizeof(int));
    cudaMalloc((void**)&d_num, sizeof(int));
    cudaMalloc((void**)&d_S, numq * numq * sizeof(cuDoubleComplex));

    // copy data from host to device
    cudaMemcpy(d_xij, xij.data(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_yij, yij.data(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_qx, qx.data(), qx.size() * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_qy, qy.data(), qy.size() * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_numseps, &numseps, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_numq, &numq, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_num, &num, sizeof(int), cudaMemcpyHostToDevice);

    int threadsPerBlock = 512;
    int blocksPerGrid = (numq * numq + threadsPerBlock - 1) / threadsPerBlock;

    // launch the kernel
    computeS<<<blocksPerGrid, threadsPerBlock>>>(d_S, d_xij, d_yij, d_qx, d_qy, d_numseps, d_numq,d_num);

    // Wait for GPU to finish before accessing on host
    cudaDeviceSynchronize();





    std::vector<cuDoubleComplex> h_S(numq * numq);
    cudaMemcpy(h_S.data(), d_S, numq * numq * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);
    for(int i = 0; i < h_S.size(); i++) std::cout << qx[i/numq]<<"\t"<<qy[i%numq]<<"\t" << cuCreal(h_S[i]) << "\t" << cuCimag(h_S[i]) << std::endl;
    
    cudaFree(d_xij);
    cudaFree(d_yij);
    cudaFree(d_qx);
    cudaFree(d_qy);
    cudaFree(d_numseps);
    cudaFree(d_S);


    //////////////////////////////////////////////////





    return 0;
}