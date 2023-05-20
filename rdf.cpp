#include "edmd_functions.hpp"

int main(int argc, char* argv[]) {
    if (argc != 2) {  // If the number of arguments is not 2 (program name + integer)
        std::cerr << "Usage: " << argv[0] << " [integer]\n";
        return 1;
    }

    int run = std::atoi(argv[1]);  // Convert the second argument to an integer


    double L = 543.1092384666489;//212.15204627603475;
    double R = 1;
    long long num = 256*256;
    double phi = 0.698;


    double dr = 0.001;
    double low = 2.0+dr; double high = 2.1;
    std::vector<double> bins = {low};
    std::vector<double> bincs = {};

    long long numpairs = 0; // Total number of pairs encountered.
        

    while(bins.back()<high-dr)
    {
        bins.push_back(bins.back()+dr);
    }
    // for(int i = 0;i<bins.size();i++) std::cout << bins[i] << std::endl;

    int numbins = bins.size();
    std::vector<int> counts(numbins-1,0);
    for(int i =0;i<numbins-1;i++) bincs.push_back( (bins[i]+bins[i+1])*0.5 );


    int ncells = static_cast<int>(std::floor(L / (2 * R)));
    double cellsize = L / ncells;
    std::vector<double> cellcenters(ncells);
    for (int i = 0; i < ncells; ++i) cellcenters[i] = -0.5 * L + 0.5 * cellsize + cellsize * i;



    for(int frame=0;frame<100000000;frame+=10000)
    {
    std::string filename = "data/pos_"+std::to_string(run)+"_" + std::to_string(frame) + ".bin";
    std::cout << filename << std::endl;
    numpairs += num*(num-1);
    std::vector<double> data = readBinaryFile(filename);
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
                if(bins[j]<=dist && dist<bins[j+1]) counts[j]+=1;
            }

        }
    }
    }
    std::vector<double> gr = {};

    double pi = 3.14159265358979;
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




