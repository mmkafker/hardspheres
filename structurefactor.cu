// Compute static structure factor, extract first Bragg peak, then use it to compute positional correlation function.
#include <iostream>
#include <fstream>
#include <complex>
#include <vector>
#include "edmd_functions.hpp"
#include <random>
#include <utility>
#include <boost/math/special_functions/bessel.hpp>
#include <cuda_runtime.h>


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

// __global__ void computeJQR(float* d_jqr, double* d_qq, double* d_allrs, int* d_numq, int* d_numrs) {
//     int index = blockIdx.x * blockDim.x + threadIdx.x;
//     long int total_size = (*d_numq); total_size*= (*d_numrs);
//     // printf("total_size = %ld\n",total_size);

//     if(index < total_size) {
//         int iq = index % (*d_numq); // row index
//         int ir = index / (*d_numq); // column index
//         d_jqr[ir * (*d_numq) + iq] = 0.0f;
//         d_jqr[ir * (*d_numq) + iq] = j0f(d_allrs[ir] * d_qq[iq]);
//         if(index==0)printf("%.8g\n",j0f(d_allrs[ir] * d_qq[iq]));
//     }
// }
// __global__ void computeJQR(float* d_jqr, double* d_qq, double* d_allrs, int* d_numq, int* d_numrs) {
//     int ir = blockIdx.x * blockDim.x + threadIdx.x;

//     if(ir < *d_numrs) {
//         for (int iq = 0; iq < *d_numq; iq++) {
//             long long index = static_cast<long long>(ir) * (*d_numq) + iq;
//             d_jqr[index] = j0f(d_allrs[ir] * d_qq[iq]);
//         }
//     }
// }


// __global__ void computeSq(double* d_Sq, float* d_jqr, int* d_numq, int* d_numrs, int* d_num) {
//     int iq = blockIdx.x * blockDim.x + threadIdx.x;

//     if(iq < *d_numq) {
//         double sum = 0;
//         for(int ir = 0; ir < *d_numrs; ++ir) {
//             sum += d_jqr[ir * (*d_numq) + iq];
//         }
//         d_Sq[iq] = (sum * 2 / *d_num) + 1;
//     }
// }

__global__ void computeJQR(int iq, double* d_jqr, double* d_qq, double* d_allrs, int* d_numq, int* d_numrs) {
    int ir = blockIdx.x * blockDim.x + threadIdx.x;

    if(ir < *d_numrs) d_jqr[ir] = j0(d_qq[iq]*d_allrs[ir]);
}

__global__ void computeSq(int iq, double* d_jqr, double* d_Sq, int* d_numrs, int* d_num) {
    double sum = 0.0;
    for (int ir = 0; ir < *d_numrs; ir++) {
        sum += d_jqr[ir];
    }
    d_Sq[iq] = 2.0 / (*d_num) * sum + 1.0;
}






int main(int argc, char* argv[]) {

    if (argc != 3) {  // If the number of arguments is not 2 (program name + integer)
        std::cerr << "Usage: " << argv[0] << " simid L \n";
        return 1;
    }

    // cudaSetDevice(1);

    int simid = std::atoi(argv[1]);  // Convert the second argument to an integer
    double L = std::atof(argv[2]);  // atof converts C-string to double

    const int num = 256*256; // replace with your number
    // std::vector<std::vector<std::complex<double>>> psisAll = readPsis("psis.bin", num);
    

    // Compute displacements for the static structure factor
    //////////////////////////////////////////////////
    int numsamplepairs = 10*num;

    std::vector<int> frames;
    for(int frame = 0;frame<34500;frame+=100) frames.push_back(frame);

    std::vector<double> allrs;

    for(int frameind = 200; frameind<frames.size();frameind++) // WARNING: WE CHOOSE THE STARTING VALUE TO GET 20,000 POSITION WRITINGS
    {
        // std::cout << frameind << std::endl;
        int frame = frames[frameind];

        std::vector<double> data = readPositionFrame(simid, frame);

        std::vector<std::vector<double>> positions(data.size()/2, std::vector<double>(2));
        for (int i = 0; i < data.size(); i++) {
            positions[i/2][i%2] = data[i];
        }

        std::vector<std::pair<int, int>> samples = generateSamples(num, numsamplepairs);

        std::vector<double> rsFrame;

        for(int i = 0; i< samples.size();i++) 
        {
            int ind1 = samples[i].first;
            int ind2 = samples[i].second;
            std::vector<double> r1 = positions[ind1];
            std::vector<double> r2 = positions[ind2];
            std::vector<double> r12 = {r2[0]-r1[0],r2[1]-r1[1]};
            PBCvec(r12,L);
            double r = std::sqrt(r12[0]*r12[0] + r12[1]*r12[1]);
            rsFrame.push_back(r);
        }
        allrs.insert(allrs.end(), rsFrame.begin(), rsFrame.end());
    }
    //////////////////////////////////////////////////
    int numrs = allrs.size();
    std::cout << "numrs: "<<numrs <<std::endl;

    double dq = 0.05;
    double qmax = 10.0;
    std::vector<double> qq;
    for (double i = 0; i <= qmax; i += dq) {
        qq.push_back(i);
    }
    int numq = qq.size();
    std::vector<double> Sq(numq, 0.0);

    // Allocate memory on the GPU for your arrays
    double* d_qq;
    double* d_Sq;
    double* d_allrs;
    double* d_jqr;
    int* d_numq;
    int* d_numrs;
    int* d_num;
    

    cudaMalloc((void**)&d_qq, numq * sizeof(double));
    cudaMalloc((void**)&d_Sq, numq * sizeof(double));
    cudaMalloc((void**)&d_allrs, numrs * sizeof(double));
    cudaMalloc((void**)&d_jqr, numrs * sizeof(double));
    cudaMalloc((void**)&d_numq, sizeof(int));
    cudaMalloc((void**)&d_numrs, sizeof(int));
    cudaMalloc((void**)&d_num, sizeof(int));

    // Copy data from the host (CPU) to the device (GPU)
    cudaMemcpy(d_qq, qq.data(), numq * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Sq, Sq.data(), numq * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_allrs, allrs.data(), numrs * sizeof(double), cudaMemcpyHostToDevice);
    // cudaMemcpy(d_jqr, jqr.data(), numq * numrs * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_numq, &numq, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_numrs, &numrs, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_num, &num, sizeof(int), cudaMemcpyHostToDevice);


    int blockSize = 512;
    // int gridSize = (numqr + blockSize - 1) / blockSize;
    int gridSize = (numrs + blockSize - 1) / blockSize;
    std::cout << "Initial gridSize: "<<gridSize <<std::endl;

    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);

    int maxThreadsPerBlock = deviceProp.maxThreadsPerBlock;
    int maxGridSize = deviceProp.maxGridSize[0];

    // If total number of threads exceeds the GPU's maximum
    if (gridSize * blockSize > maxThreadsPerBlock * maxGridSize) {
        // Adjust blockSize and gridSize
        blockSize = maxThreadsPerBlock;
        gridSize = (numrs + blockSize - 1) / blockSize;

        if (gridSize > maxGridSize) {
            std::cerr << "Grid size is still larger than the GPU can handle, even after adjustment" << std::endl;
            return;
        }
    }

    std::cout << "Adjusted gridSize: "<<gridSize <<std::endl;

    for (int iq = 0; iq < numq; iq++) {
        std::cout << iq <<std::endl;
        // Call the computeJQR kernel
        computeJQR<<<gridSize, blockSize>>>(iq, d_jqr, d_qq, d_allrs, d_numq, d_numrs);
        // Synchronize to make sure computeJQR has completed
        cudaDeviceSynchronize();

        // Call the computeSq kernel
        computeSq<<<1, 1>>>(iq, d_jqr, d_Sq, d_numrs, d_num);
        // Synchronize to make sure computeSq has completed
        cudaDeviceSynchronize();
    }


    // cudaError_t cudaStatus;

    // std::cout << "Computing j0(q_i r_j)..." <<std::endl;
    // computeJQR<<<gridSize, blockSize>>>(d_jqr, d_qq, d_allrs, d_numq, d_numrs);
    // cudaDeviceSynchronize();
    // std::cout << "Finished." <<std::endl;

    // cudaStatus = cudaGetLastError();

    // if (cudaStatus != cudaSuccess) {
    //     std::cerr << "Kernel launch failed: " << cudaGetErrorString(cudaStatus) << std::endl;
    //     return;
    // }


    // std::cout << "Computing S(q)..." <<std::endl;
    // gridSize = (numq + blockSize - 1) / blockSize;
    // std::cout << "gridSize: "<<gridSize <<std::endl;
    // computeSq<<<gridSize, blockSize>>>(d_Sq, d_jqr, d_numq, d_numrs, d_num);
    // cudaDeviceSynchronize();
    // std::cout << "Finished." <<std::endl;


    cudaMemcpy(Sq.data(), d_Sq, numq * sizeof(double), cudaMemcpyDeviceToHost);
    for(int i = 0;i<numq;i++) std::cout << qq[i]<<"\t"<< Sq[i]<<std::endl;


    cudaFree(d_qq);
    cudaFree(d_Sq);
    cudaFree(d_allrs);
    cudaFree(d_numq);
    cudaFree(d_numrs);
    cudaFree(d_num);
    cudaFree(d_jqr);



    





    return 0;
}


// CUDA Kernel
// __global__ void computeS(cuDoubleComplex* d_S, const double* d_xij, const double* d_yij, const double* d_qx, const double* d_qy, const int* d_numseps, const int* d_numq,const int* d_num) {
//     int iq = threadIdx.x + blockIdx.x * blockDim.x;

//     // Check if index is within range
//     if(iq >= (*d_numq) * (*d_numq)) return;

//     // Calculate qx and qy based on thread index
//     double qx = d_qx[iq/(*d_numq)];
//     double qy = d_qy[iq%(*d_numq)];

//     // Initialize sum as a complex number with zero real and imaginary parts
//     cuDoubleComplex sum = make_cuDoubleComplex(0.0, 0.0);

//     for (int i = 0; i < *d_numseps; i++) {

//         if(i%1000000 ==0) printf("iq = %d, i = %d of %d\n",iq,i,*d_numseps);
//         // Compute e^(i(qx*xij + qy*yij))
//         double exponent = qx * d_xij[i] + qy * d_yij[i];
//         cuDoubleComplex value = make_cuDoubleComplex(cos(exponent), sin(exponent));

//         // Add value to sum
//         sum = cuCadd(sum, value);
//     }

//     // Scale the sum by 1/numseps
//     sum = make_cuDoubleComplex(1+cuCreal(sum) / (*d_num), cuCimag(sum) / (*d_num));

//     // Store the result in the d_S array
//     d_S[iq] = sum;
// }


// Compute static structure factor on the GPU
    //////////////////////////////////////////////////
    // std::vector<double> xij;
    // std::vector<double> yij;

    // for (const auto& sep : allSeps) {
    //     xij.push_back(sep[0]);
    //     yij.push_back(sep[1]);
    // }

    // std::vector<double> qx, qy;
    // for (double i = -10; i <= 10; i += 0.5) {
    //     qx.push_back(i);
    //     qy.push_back(i);
    // }
    // int numseps = xij.size();
    // int numq = qx.size();

    // double* d_xij;
    // double* d_yij;
    // double* d_qx;
    // double* d_qy;
    // int* d_numseps;
    // int* d_numq;
    // int* d_num;
    // cuDoubleComplex* d_S;
    

    // int size = numseps * sizeof(double);

    // // allocate GPU memory
    // cudaMalloc((void**)&d_xij, size);
    // cudaMalloc((void**)&d_yij, size);
    // cudaMalloc((void**)&d_qx, qx.size() * sizeof(double));
    // cudaMalloc((void**)&d_qy, qy.size() * sizeof(double));
    // cudaMalloc((void**)&d_numseps, sizeof(int));
    // cudaMalloc((void**)&d_numq, sizeof(int));
    // cudaMalloc((void**)&d_num, sizeof(int));
    // cudaMalloc((void**)&d_S, numq * numq * sizeof(cuDoubleComplex));

    // // copy data from host to device
    // cudaMemcpy(d_xij, xij.data(), size, cudaMemcpyHostToDevice);
    // cudaMemcpy(d_yij, yij.data(), size, cudaMemcpyHostToDevice);
    // cudaMemcpy(d_qx, qx.data(), qx.size() * sizeof(double), cudaMemcpyHostToDevice);
    // cudaMemcpy(d_qy, qy.data(), qy.size() * sizeof(double), cudaMemcpyHostToDevice);
    // cudaMemcpy(d_numseps, &numseps, sizeof(int), cudaMemcpyHostToDevice);
    // cudaMemcpy(d_numq, &numq, sizeof(int), cudaMemcpyHostToDevice);
    // cudaMemcpy(d_num, &num, sizeof(int), cudaMemcpyHostToDevice);

    // int threadsPerBlock = 512;
    // int blocksPerGrid = (numq * numq + threadsPerBlock - 1) / threadsPerBlock;

    // // launch the kernel
    // computeS<<<blocksPerGrid, threadsPerBlock>>>(d_S, d_xij, d_yij, d_qx, d_qy, d_numseps, d_numq,d_num);

    // // Wait for GPU to finish before accessing on host
    // cudaDeviceSynchronize();





    // std::vector<cuDoubleComplex> h_S(numq * numq);
    // cudaMemcpy(h_S.data(), d_S, numq * numq * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);
    // for(int i = 0; i < h_S.size(); i++) std::cout << qx[i/numq]<<"\t"<<qy[i%numq]<<"\t" << cuCreal(h_S[i]) << "\t" << cuCimag(h_S[i]) << std::endl;
    
    // cudaFree(d_xij);
    // cudaFree(d_yij);
    // cudaFree(d_qx);
    // cudaFree(d_qy);
    // cudaFree(d_numseps);
    // cudaFree(d_S);


    //////////////////////////////////////////////////

