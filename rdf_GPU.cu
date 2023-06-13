#include <stdio.h>
#include <cuda_runtime.h>
#include <stdlib.h>
#include <vector>
#include "edmd_functions.hpp"

std::vector<int> flattenCell2Particle(const std::unordered_map<std::pair<int, int>, std::vector<int>>& cell2particle, int ncells, double cellsize) {
    // Calculate the length of each cell stretch in the flat vector
    int lengthpercell = std::ceil(cellsize * cellsize / (2 * std::sqrt(3)));

    // Initialize the flat vector with -2
    std::vector<int> cell2particleflat(ncells * ncells * lengthpercell, -2);

    // Loop over all pairs in the unordered_map
    for (const auto& pair : cell2particle) {
        // Calculate the index in the flat vector using column-major ordering
        int idx = (pair.first.first + pair.first.second * ncells) * lengthpercell;

        // Replace the entries of the flat vector with the values from the unordered_map
        for (size_t i = 0; i < pair.second.size(); ++i) {
            cell2particleflat[idx + i] = pair.second[i];
        }
    }

    return cell2particleflat;
}

std::vector<int> flattenParticle2Cell(const std::vector<std::vector<int>>& particle2cell, int ncells)
{
    std::vector<int> particle2cellflat(particle2cell.size());
    
    for(int i = 0; i < particle2cell.size(); ++i)
    {
        // Take the {i, j} pair and convert it to a column-major index
        particle2cellflat[i] = particle2cell[i][0] + particle2cell[i][1]*ncells;
    }

    return particle2cellflat;
}

// This CUDA device function corrects a position component using periodic boundary conditions.
__device__ void PBCvec_GPU(double* x, double *L)
{
    for (int i = 0; i < 2; ++i) 
    {
        if (abs(x[i]) >= 0.5 * (*L)) 
        {
            x[i] -= (*L) * ((x[i] > 0) ? 1 : -1);
        }
    }
}

__device__ int cmod_GPU(int x, int N) 
{
    int result = x % N;
    return result < 0 ? result + N : result;
}

__device__ void computeNeighborIndices(int* neighinds, int pcell, int ncells)
{
    // Convert the cell index back to a 2D index
    int c0 = pcell % ncells; 
    int c1 = pcell / ncells; 

    // Compute modified indices
    int c0p1 = cmod_GPU((c0+1),ncells);
    int c0m1 = cmod_GPU((c0-1),ncells);
    int c1p1 = cmod_GPU((c1+1),ncells);
    int c1m1 = cmod_GPU((c1-1),ncells);

    // Construct the neighbor indices using column-major ordering
    neighinds[0] = c0 + c1 * ncells;
    neighinds[1] = c0 + c1m1 * ncells;
    neighinds[2] = c0 + c1p1 * ncells;
    neighinds[3] = c0m1 + c1 * ncells;
    neighinds[4] = c0m1 + c1m1 * ncells;
    neighinds[5] = c0m1 + c1p1 * ncells;
    neighinds[6] = c0p1 + c1 * ncells;
    neighinds[7] = c0p1 + c1m1 * ncells;
    neighinds[8] = c0p1 + c1p1 * ncells;
}

__device__ void getNeighborParticles(int* neighbors, int* cell2particleflat, const int* neighinds, int p, int lengthpercell, int ncells)
{
    // Initialize neighbors to -2
    for(int i = 0; i < 9 * lengthpercell; ++i) {
        neighbors[i] = -2;
    }

    // Temporary index to keep track of where to insert in neighbors
    int idx = 0;

    // Loop over neighboring cells
    for(int i = 0; i < 9; ++i) {
        int cell_start = neighinds[i] * lengthpercell; // start index of cell in cell2particleflat
        int cell_end = cell_start + lengthpercell; // end index of cell in cell2particleflat
        
        // Loop over particles in cell
        for(int j = cell_start; j < cell_end; ++j) {
            int particle = cell2particleflat[j];

            // Only add particle if it's not -2 and it's not p
            if(particle >= 0 && particle != p) {
                neighbors[idx] = particle;
                idx++;
            }
        }
    }
}

__device__ void computeSPHistogram(double* d_xs, double* d_ys, double *d_L, int p, int* neighbors, double* d_bins, long long offset, int* d_hist, int *d_numbins, int *d_lengthpercell)
{
    double rij[2];

    // Loop over neighbors
    for(int i = 0; i < 9 * (*d_lengthpercell); ++i) {
        int neighbor = neighbors[i];

        // Only process valid neighbors
        if(neighbor >= 0) {
            rij[0] = d_xs[p] - d_xs[neighbor];
            rij[1] = d_ys[p] - d_ys[neighbor];
            
            PBCvec_GPU(rij, d_L);
            double dist = sqrt(rij[0]*rij[0] + rij[1]*rij[1]);

            // Check bins and increment counts
            for(int j = 0; j < (*d_numbins) - 1; ++j) {
                if(d_bins[j] <= dist && dist < d_bins[j+1]) {
                    d_hist[offset + j] += 1;
                }
            }
        }
    }
}

__global__ void compHistFrame(double* d_xs, double* d_ys, double *d_L, int* d_particle2cellflat, int* d_cell2particleflat, double* d_bins, int* d_hist, int *d_numbins, int *d_lengthpercell, int *ncells, long long *num, int *d_neighbors)
{
    // Calculate the global index
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    // Ensure we do not go out of bounds
    if(idx < *num) {
        // Calculate offset for this thread
        long long offset = ((*d_numbins)-1) * idx;

        // Compute neighbor indices
        int neighinds[9];
        computeNeighborIndices(neighinds, d_particle2cellflat[idx], *ncells);

        // Compute neighbors
        
        getNeighborParticles(d_neighbors, d_cell2particleflat, neighinds, idx, *d_lengthpercell, *ncells);

        // Call the device function to compute the histogram
        computeSPHistogram(d_xs, d_ys, d_L, idx, d_neighbors, d_bins, offset, d_hist, d_numbins, d_lengthpercell);
    }
}

__global__ void collapseHist(int* d_hist, int* d_counts, int *numbins, int *d_total_frames)
{
    // Loop over bins
    for(int i = 0; i < (*numbins)-1; ++i) 
    {
        // Accumulate over frames
        for(int j = 0; j < (*d_total_frames); ++j) 
        {
            d_counts[i] += d_hist[j*((*numbins)-1) + i];
        }
    }
}






int main(int argc, char* argv[]) 
{

    if (argc != 3) {  // If the number of arguments is not 2 (program name + integer)
        std::cerr << "Usage: " << argv[0] << " simid L \n";
        return 1;
    }

    int simid = std::atoi(argv[1]);  // Convert the second argument to an integer

    double L = std::atof(argv[2]);  // atof converts C-string to double

    double R = 1;
    long long num = 256*256;
    double pi = 3.14159265358979;
    double phi = num*pi/(L*L);
    // std::cout << "phi = " << phi << std::endl;
    // double phi = 0.698;


    double dr = 0.001; // To get RDF at contact
    // double dr = 0.1; // To get RDF farther out
    double low = 2.0+dr; 
    double high = 2.1; // To get RDF at contact
    // double high = 10.0; // To get RDF farther out
    std::vector<double> bins = {low};
    std::vector<double> bincs = {};

    long long numpairs = 0; // Total number of pairs encountered.
        

    while(bins.back()<high-dr) bins.push_back(bins.back()+dr);
    

    int numbins = bins.size();
    std::vector<int> counts(numbins-1,0);
    for(int i =0;i<numbins-1;i++) bincs.push_back( (bins[i]+bins[i+1])*0.5 );


    int ncells = static_cast<int>(std::floor(L / (2 * R))); // To get RDF at contact
    // int ncells = static_cast<int>(std::floor(L / (10 * R)));// To get RDF farther out
    double cellsize = L / ncells;
    std::vector<double> cellcenters(ncells);
    for (int i = 0; i < ncells; ++i) cellcenters[i] = -0.5 * L + 0.5 * cellsize + cellsize * i;

    std::string filename = "pos_"+std::to_string(simid)+".bin";

    const int start_frame = 20000;
    const int end_frame = 20010;
    int total_frames = end_frame - start_frame;
    int gpuhistsize = total_frames * (numbins-1);
    std::vector<int> hist(gpuhistsize,0);
    int lengthpercell = std::ceil(cellsize * cellsize / (2 * std::sqrt(3)));


    int threadsPerBlock = 512;
    int numBlocks = num/128;

    // Allocate memory on the GPU
    int* d_cell2particleflat; 
    int* d_particle2cellflat;
    double* d_xs;
    double* d_ys;
    double* d_L;
    double* d_bins;
    int* d_numbins;
    int* d_lengthpercell;
    int* d_hist;
    int* d_ncells;
    long long* d_num;
    int* d_counts;
    int* d_total_frames;
    int* d_neighbors; //[9 * (*d_lengthpercell)];
    
    
    for(int frame=start_frame;frame<end_frame;frame++)
    {
        std::vector<double> data = readPositionFrame(simid, frame);
        numpairs += num*(num-1);

        // std::vector<double> data = readBinaryFile(filename);
        std::vector<std::vector<double>> positions(data.size()/2, std::vector<double>(2));
        for (int i = 0; i < data.size(); i++) {
            positions[i/2][i%2] = data[i];
        }

        if(positions.size() != num) 
        {
            std::cout << "Check value of 'num'\n";
            std::cout << "positions.size() = "<<positions.size()<<", num = "<<num <<std::endl;
            assert(false);
        }

        std::vector<std::vector<int>> particle2cell = AssignCells(cellcenters, positions, num);
        std::unordered_map<std::pair<int, int>, std::vector<int>> cell2particle = GenCell2Particle(particle2cell, num, ncells);

        std::vector<int> cell2particleflat = flattenCell2Particle(cell2particle, ncells, cellsize);
        std::vector<int> particle2cellflat = flattenParticle2Cell(particle2cell, ncells);

        std::vector<double> xs(num); std::vector<double> ys(num); 

        for(int i = 0;i<num;i++)
        {
            xs[i] = positions[i][0];
            ys[i] = positions[i][1];
        }

        if (frame==start_frame)
        {
            cudaMalloc((void**)&d_cell2particleflat, cell2particleflat.size() * sizeof(int));
            cudaMalloc((void**)&d_particle2cellflat, particle2cellflat.size() * sizeof(int));
            cudaMalloc((void**)&d_xs, xs.size() * sizeof(double));
            cudaMalloc((void**)&d_ys, ys.size() * sizeof(double));
            cudaMalloc((void**)&d_bins, bins.size() * sizeof(double));
            cudaMalloc((void**)&d_L, sizeof(double));
            cudaMalloc((void**)&d_numbins, sizeof(int));
            cudaMalloc((void**)&d_lengthpercell, sizeof(int));
            cudaMalloc((void**)&d_hist, gpuhistsize * sizeof(int));
            cudaMalloc((void**)&d_ncells, sizeof(int));
            cudaMalloc((void**)&d_num, sizeof(long long));
            cudaMalloc((void**)&d_counts, counts.size() * sizeof(int));
            cudaMalloc((void**)&d_total_frames, sizeof(int));
            cudaMalloc((void**)&d_neighbors, 9*lengthpercell*sizeof(int));
        }

        

        // Copy data from host to device
        cudaMemcpy(d_cell2particleflat, cell2particleflat.data(), cell2particleflat.size() * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_particle2cellflat, particle2cellflat.data(), particle2cellflat.size() * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_xs, xs.data(), xs.size() * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_ys, ys.data(), ys.size() * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_bins, bins.data(), bins.size() * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_L, &L, sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_numbins, &numbins, sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_lengthpercell, &lengthpercell, sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_hist, hist.data(), hist.size() * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_ncells, &ncells, sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_num, &num, sizeof(long long), cudaMemcpyHostToDevice);
        cudaMemcpy(d_counts, counts.data(), counts.size() * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_total_frames, &total_frames, sizeof(int), cudaMemcpyHostToDevice);

        // Call the kernel function
        compHistFrame<<<numBlocks, threadsPerBlock>>>(d_xs, d_ys, d_L, d_particle2cellflat, d_cell2particleflat, d_bins, d_hist, d_numbins, d_lengthpercell, d_ncells, d_num,d_neighbors);

    }
    std::vector<int> longhist((numbins-1)*num);
    cudaMemcpy(longhist.data(), d_hist, longhist.size() * sizeof(int), cudaMemcpyDeviceToHost);
    
    // collapseHist<<<1,1>>>(d_hist,d_counts, d_numbins, d_total_frames);


    // cudaMemcpy(counts.data(), d_counts, counts.size() * sizeof(int), cudaMemcpyDeviceToHost);

    for(int i = 0; i< longhist.size();i++) std::cout << longhist[i] << std::endl;







    cudaFree(d_cell2particleflat); 
    cudaFree(d_particle2cellflat);
    cudaFree(d_xs);
    cudaFree(d_ys);
    cudaFree(d_L);
    cudaFree(d_bins);
    cudaFree(d_numbins);
    cudaFree(d_lengthpercell);
    cudaFree(d_hist);
    cudaFree(d_ncells);
    cudaFree(d_num);
    cudaFree(d_counts);
    cudaFree(d_total_frames);

}

