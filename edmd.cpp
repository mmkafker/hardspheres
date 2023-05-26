#include "edmd_functions.hpp"


int main(int argc, char** argv) {

    if (argc != 5) {
        std::cerr << "Usage: " << argv[0] << " <integer> <double> <filename> <steps>" << std::endl;
        return 1; // return with error code
    }
    
    // Argument 1 - Integer
    int simid = std::stoi(argv[1]);

    // Argument 2 - Double
    double L = std::atof(argv[2]);  // atof converts C-string to double

    // Argument 3 - Filename
    std::string inputFile = argv[3];

    // Argument 4 – Steps
    long long steps = std::stoll(argv[4]);



    
    double R = 1;
    int num = 256*256;

    std::vector<double> data = readBinaryFile(inputFile);

    // Assuming the data should be reshaped into a 2D array with two columns
    std::vector<std::vector<double>> IPs(data.size()/2, std::vector<double>(2));
    for (int i = 0; i < data.size(); i++) {
        IPs[i/2][i%2] = data[i];
    }

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-1, 1);

    std::vector<std::vector<double>> IVs(num, std::vector<double>(2));
    for (int i = 0; i < num; i++) {
        IVs[i][0] = dis(gen);
        IVs[i][1] = dis(gen);
    }

    std::vector<double> pi(2, 0.0);
    std::vector<double> pf(2, 0.0);
    double Ei = 0.0;
    double Ef = 0.0;

    for (int p = 0; p < num; p++) {
        pi[0] += IVs[p][0];
        pi[1] += IVs[p][1];
        Ei += 0.5 * (IVs[p][0]*IVs[p][0] +  IVs[p][1]*IVs[p][1]);
    }

    std::string filename = "output_" + std::to_string(simid) + ".txt";
    std::ofstream outfile(filename, std::ios::app);
    if (outfile.is_open())
    {
        outfile << "Energy "<<Ei << std::endl;    
        outfile << "L "<<L << std::endl;    
        outfile << "Input file "<<inputFile << std::endl;    
        outfile << "Number of particles "<<num << std::endl;    
        outfile << "Number of steps "<<steps << std::endl;    
        outfile.close();
    }
    else
    {
        std::cout << "Unable to open file";
    }
    std::vector<std::vector<double>> state(num, std::vector<double>(5, 0.0));
    std::vector<std::tuple<int, int, double>> collisions = {};

    long long substeps = 5000000;

    long long checkpointid = 0;
    while (checkpointid*substeps<steps)
    {
        
        std::cout << "Checkpoint " << checkpointid << std::endl;
        HardSphereEDMD(IPs, IVs, L, R, num, substeps,state,collisions,simid,checkpointid);
        std::vector<std::vector<double>> statetf = printstate(state,L);

        writeState(statetf, simid, checkpointid);

        writeCollisions(collisions, simid,checkpointid);
        collisions.clear();

        std::string filename = "output_" + std::to_string(simid) + ".txt";
        std::ofstream outfile(filename, std::ios::app);
        if (outfile.is_open())
        {
            outfile << "Checkpoint "<<checkpointid << " time = " << statetf[0][0] << std::endl;    
            outfile.close();
        }
        else
        {
            std::cout << "Unable to open output file";
        }


        for(int i = 0;i<num;i++)
        {
            IPs[i][0] = statetf[i][1];
            IPs[i][1] = statetf[i][2];
            IVs[i][0] = statetf[i][3];
            IVs[i][1] = statetf[i][4];
        }

        checkpointid++;
    }

    for(int i = 0;i<num;i++)
    { 
        Ef+= 0.5*(state[i][3]*state[i][3] + state[i][4]*state[i][4]);
        pf[0] += state[i][3];
        pf[1] += state[i][4];
    }
    std::cout << "Final energy " << Ef << std::endl;
    std::cout << "Final momentum (" << pf[0] <<", " << pf[1] <<")" << std::endl;

    // writeCollisions(collisions, simid);




    std::cout << "Momentum conservation: " << (pi[0]-pf[0])/pi[0] << " " << (pi[1]-pf[1])/pi[1] << std::endl;
    std::cout << "Energy conservation: " << std::abs((Ei - Ef) / Ei) << std::endl;
    // std::cout << "Final energy = " << Ef << std::endl;

    return 0;
}




