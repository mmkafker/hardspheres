#include "edmd_functions.hpp"


int main(int argc, char** argv) {
    if(argc < 2) {
        std::cerr << "You must provide a single integer as argument.\n";
        return 1; // Return with error code
    }

    int simid = std::atoi(argv[1]);

    double L = 543.1092384666489;//212.15204627603475;
    double R = 1;
    int num = 256*256;
    int steps = 500;

    std::vector<double> data = readBinaryFile("disorderedIPs256.bin");

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
    // std::cout << "Initial energy " << Ei << std::endl;
    // std::cout << "Initial momentum (" << pi[0] <<", " << pi[1] <<")" << std::endl;

    std::vector<std::vector<double>> state(num, std::vector<double>(5, 0.0));
    std::vector<std::tuple<int, int, double>> collisions = {};
    HardSphereEDMD(IPs, IVs, L, R, num, steps,state,collisions,simid);

    for(int i = 0;i<num;i++)
    { 
        Ef+= 0.5*(state[i][3]*state[i][3] + state[i][4]*state[i][4]);
        pf[0] += state[i][3];
        pf[1] += state[i][4];
    }
    std::cout << "Final energy " << Ef << std::endl;
    std::cout << "Final momentum (" << pf[0] <<", " << pf[1] <<")" << std::endl;


    // std::vector<std::vector<double>> statetf = printstate(state,L);
    // writePositions(statetf, "writertest.bin");

    //for(int i=0;i<num;i++) std::cout<<std::get<0>(collisions[i])<<" "<<std::get<1>(collisions[i])<<" "<<std::get<2>(collisions[i])<<std::endl;
    std::string filename = "collisions_" + std::to_string(simid) + ".txt";
    std::ofstream outfile(filename);

    if (outfile.is_open())
    {
        for(int i=0;i<num;i++)
            outfile << std::get<0>(collisions[i]) << "\t"
                    << std::get<1>(collisions[i]) << "\t"
                    << std::get<2>(collisions[i]) << std::endl;

        outfile.close();
    }
    else
    {
        std::cout << "Unable to open file";
    }

    filename = "energy_" + std::to_string(simid) + ".txt";
    outfile.open(filename);

    if (outfile.is_open())
    {
        outfile << Ef << std::endl;    
        outfile.close();
    }
    else
    {
        std::cout << "Unable to open file";
    }

    std::cout << "Momentum conservation: " << (pi[0]-pf[0])/pi[0] << " " << (pi[1]-pf[1])/pi[1] << std::endl;
    std::cout << "Energy conservation: " << std::abs((Ei - Ef) / Ei) << std::endl;
    // std::cout << "Final energy = " << Ef << std::endl;

    return 0;
}




