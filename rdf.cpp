// This implements a threaded RDF calculation, which can compute the RDF at contact for 256*256 particles for 20,000 
// steps in 25 minutes by multithreading on the CPU, where the parallelism is done over separate time steps, and communication is used
// to update a shared pairwise distance histogram.

// If we want to make histograms over larger distances (say, cell sizes of 5 diameters, the execution time is 10 hours) for
// the same number of steps, which is unacceptable given my current constraints.



#include "edmd_functions.hpp"
#include <thread>
#include <mutex>

std::mutex mtx_numpairs; // mutex for critical section
std::mutex mtx_counts; // mutex for critical section
std::mutex mtx_print;

// Function to handle work of a single frame
void handle_frame_threaded(int frame, long long& numpairs, std::vector<int>& counts,int simid, long long num, std::vector<double> &cellcenters, int ncells,
            double L,int numbins, std::vector<double> &bins)
{
    // {
    //     std::lock_guard<std::mutex> guard(mtx_print);
    //     std::cout << frame << std::endl;
    // }
    std::vector<double> data = readPositionFrame(simid, frame);

    {
        std::lock_guard<std::mutex> guard(mtx_numpairs); // Locks mtx during this scope
        numpairs += num*(num-1);
    }

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

        for (int i = 0; i < neighbors.size(); i++) 
        {
            std::vector<double> rij = {positions[p][0]-positions[neighbors[i]][0],positions[p][1]-positions[neighbors[i]][1]};
            PBCvec(rij,L);
            double dist = std::sqrt(rij[0]*rij[0] + rij[1]*rij[1]);
            for(int j = 0;j<numbins-1;j++)
            {
                if(bins[j]<=dist && dist<bins[j+1]) 
                {
                    // Critical section for counts
                    {
                        std::lock_guard<std::mutex> guard(mtx_counts);
                        counts[j]+=1;
                    }
                }
            }

        }
    }
}

int main(int argc, char* argv[]) {
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
    const int end_frame = 20063;
    const int num_threads = 64;

    std::vector<std::thread> threads(num_threads);
    int frames_per_thread = (end_frame - start_frame) / num_threads;
    int remaining_frames = (end_frame - start_frame) % num_threads;

    for(int i = 0; i < num_threads; i++)
    {
        int extra_frames = 0; // Extra frames for threads [0, remaining_frames)
        for(int j = 0; j < i; j++)
        {
            if (j < remaining_frames)
                extra_frames++;
        }

        int begin = start_frame + i * frames_per_thread + extra_frames;
        int end = begin + frames_per_thread + (i < remaining_frames ? 1 : 0);

        threads[i] = std::thread([begin, end, &numpairs, &counts, &simid, &num, &cellcenters, &ncells, &L, &numbins, &bins]()
        {
            for(int frame = begin; frame < end; frame++)
            {
                {
                    std::lock_guard<std::mutex> guard(mtx_print);
                    std::cout << frame << std::endl;
                }
                handle_frame_threaded(frame, numpairs, counts, simid, num, cellcenters, ncells, L, numbins, bins);
            }
        });
    }

    // Join the threads
    for(auto& thread : threads)
    {
        thread.join();
    }


    std::vector<double> gr = {};

    
    for(int i =0;i<numbins-1;i++)
    {
        // std::cout << "1: "<< (1.0*num*pi/phi)*counts[i] << std::endl;
        // std::cout << "2: "<<"pi = "<<pi <<", bincs[i] = "<<bincs[i]<<", dr = "<<dr<<", numpairs = "<<numpairs<<", combo = "<<(2.0*pi*bincs[i]*dr*numpairs) << std::endl;

        gr.push_back(  (1.0*num*pi/phi)*counts[i]/(2.0*pi*bincs[i]*dr*numpairs)  );
    } 

    for(int i =0;i<numbins-1;i++)
    {
        std::cout << bincs[i]<<"\t"<<gr[i] << std::endl;
    }


    return 0;
}




