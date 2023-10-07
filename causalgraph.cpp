//g++ -std=c++17 -pthread causalgraph.cpp -o xcg

// This code constructs the causal graph from the list of collisions and then computes stuff with it.

#include <fstream>
#include <vector>
#include <tuple>
#include <sstream>
#include <iostream>
#include <string>
#include <algorithm>
#include <thread>
#include <mutex>
#include <queue>
#include <unordered_map>
#include <unordered_set>
#include <chrono>
#include <cmath>

std::mutex mtx;

std::vector<std::tuple<int, int, double>> readCollisions(const std::string& filename) {
    std::ifstream file(filename);
    if (!file) {
        throw std::runtime_error("Unable to open file " + filename);
    }

    std::vector<std::tuple<int, int, double>> data;
    std::string line;
    while (std::getline(file, line)) {
        std::istringstream iss(line);
        int p1, p2;
        double t;
        if (!(iss >> p1 >> p2 >> t)) {
            throw std::runtime_error("Error reading line: " + line);
        }
        data.push_back(std::make_tuple(p1, p2, t));
    }
    return data;
}

void getSPCollisions(int start, int end, const std::vector<std::tuple<int, int, double>>& collisions, std::vector<std::tuple<int, int>>& edgelist) {
    for (int p = start; p < end; ++p) {
        std::vector<int> collsp;
        
        for (int i = 0; i < collisions.size(); ++i) {
            if (std::get<0>(collisions[i]) == p || std::get<1>(collisions[i]) == p) {
                collsp.push_back(i);
            }
        }
        // std::cout << "collsp.size() = "<<collsp.size() << std::endl;
        
        mtx.lock();
        if (collsp.size()>0)
        {
            for (int i = 0; i < collsp.size() - 1; ++i) {
                edgelist.push_back(std::make_tuple(collsp[i], collsp[i + 1]));
            }
        }
        mtx.unlock();
    }
}


std::vector<std::tuple<int, int>> genCausalGraph(const std::vector<std::tuple<int, int, double>>& collisions, long long num, int numthreads) {
    std::vector<std::tuple<int, int>> edgelist;
    std::vector<std::thread> threads;

    long long chunkSize = num / numthreads;
    long long remainder = num % numthreads;
    // std::cout << "chunkSize: "<< chunkSize << std::endl;
    // std::cout << "remainder: "<< remainder << std::endl;
    

    for (int t = 0; t < numthreads; ++t) {
        int extra = 0; // Extra tasks for threads [0, remainder)
        for(int j = 0; j < t; ++j)
        {
            if (j < remainder)
                extra++;
        }

        long long start = t * chunkSize + extra;
        long long end = start + chunkSize + (t < remainder ? 1 : 0);
        // std::cout << "t = " << t <<", start = "<<start <<", end = "<<end <<std::endl;

        threads.push_back(std::thread(getSPCollisions, start, end, std::ref(collisions), std::ref(edgelist)));
    }

    for (auto& th : threads) {
        th.join();
    }
    
    return edgelist;
}

void writeCausalGraphToFile(const std::vector<std::tuple<int, int>>& edgelist, const std::string& filename) {
  // Open the file for writing.
  std::ofstream file(filename, std::ios::binary);
  if (!file.is_open()) {
    std::cerr << "Error opening file '" << filename << "'" << std::endl;
    return;
  }

  // Write each edge to the file.
  for (const auto& edge : edgelist) {
    int from, to;
    std::tie(from, to) = edge;
    file.write((char*)&from, sizeof(from));
    file.write((char*)&to, sizeof(to));
  }

  // Close the file.
  file.close();
}

std::vector<std::tuple<int, int>> readCausalGraphFromFile(const std::string& filename) {
  // Open the file for reading.
  std::ifstream file(filename, std::ios::binary);
  if (!file.is_open()) {
    std::cerr << "Error opening file '" << filename << "'" << std::endl;
    return {};
  }

  // Create an empty vector to store the edges.
  std::vector<std::tuple<int, int>> edgelist;

  // Read each edge from the file and append it to the vector.
  while (file.good()) {
    int from, to;
    file.read((char*)&from, sizeof(from));
    file.read((char*)&to, sizeof(to));
    edgelist.push_back(std::make_tuple(from, to));
  }

  // Close the file.
  file.close();

  // Return the vector of edges.
  return edgelist;
}

std::unordered_map<int, std::vector<int>> createAdjacencyMap(const std::vector<std::tuple<int, int>>& edgelist, long long num_nodes) {
    std::unordered_map<int, std::vector<int>> adjacencyMap;

    // Initialize the map with all nodes
    for (long long i = 0; i < num_nodes; i++) {
        adjacencyMap[i] = std::vector<int>{};
    }

    // Populate the adjacency list from the edge list
    for (const auto& edge : edgelist) {
        int src = std::get<0>(edge);
        int dest = std::get<1>(edge);
        adjacencyMap[src].push_back(dest);
    }

    return adjacencyMap;
}


int computeSingleBallVolume(std::unordered_map<int, std::vector<int>>& adjacencyMap, int r, long long x) {
    std::queue<long long> nqueue;
    std::unordered_map<long long, long long> ndepths;
    std::unordered_set<long long> nqueue_elements;

    nqueue.push(x);
    nqueue_elements.insert(x);
    ndepths[x] = 0;

    int ballvol = 1;
    while (!nqueue.empty()) 
    {
        long long y = nqueue.front();
        nqueue.pop();
        nqueue_elements.erase(y);

        auto neighs = adjacencyMap[y];

        for (const auto& n : neighs) 
        {
            if (ndepths.find(n) == ndepths.end()) {
                ndepths[n] = ndepths[y] + 1;

                if (ndepths[n] <= r) {
                    ballvol++;

                    if (nqueue_elements.find(n) == nqueue_elements.end()) {
                        nqueue.push(n);
                        nqueue_elements.insert(n);
                    }
                }
            }
        }
    }

    return ballvol;
}

// std::vector<int> computeBallVolumes(const std::vector<std::tuple<int, int>>& edgelist, int r, long long num_nodes) {
//     std::vector<int> ballvolumes(num_nodes);
//     std::unordered_map<int, std::vector<int>> adjacencyMap = createAdjacencyMap(edgelist,num_nodes);

//     for (long long x = 0; x < num_nodes; x++) {
//         ballvolumes[x] = computeSingleBallVolume(adjacencyMap, r, x);
//     }

//     return ballvolumes;
// }

void computeChunkBallVolumes(int start, int end, std::unordered_map<int, std::vector<int>>& adjacencyMap, int r, std::vector<int>& ballvolumes) {
    for (long long x = start; x < end; x++) {
        ballvolumes[x] = computeSingleBallVolume(adjacencyMap, r, x);
    }
}

std::vector<int> computeBallVolumes(const std::vector<std::tuple<int, int>>& edgelist, int r, long long num_nodes, int numthreads) {
    std::vector<int> ballvolumes(num_nodes);
    std::unordered_map<int, std::vector<int>> adjacencyMap = createAdjacencyMap(edgelist,num_nodes);

    std::vector<std::thread> threads;
    long long chunkSize = num_nodes / numthreads;
    long long remainder = num_nodes % numthreads;

    for (int t = 0; t < numthreads; ++t) {
        int extra = 0;
        for(int j = 0; j < t; ++j)
        {
            if (j < remainder)
                extra++;
        }

        long long start = t * chunkSize + extra;
        long long end = start + chunkSize + (t < remainder ? 1 : 0);

        threads.push_back(std::thread(computeChunkBallVolumes, start, end, std::ref(adjacencyMap), r, std::ref(ballvolumes)));
    }

    for (auto& th : threads) {
        th.join();
    }

    return ballvolumes;
}









std::vector<int> getNodesAtDistance(std::unordered_map<int, std::vector<int>>& adjacencyMap, int s, long long x) {
    std::queue<long long> nqueue;
    std::unordered_map<long long, long long> ndepths;
    std::unordered_set<long long> nqueue_elements;

    nqueue.push(x);
    nqueue_elements.insert(x);
    ndepths[x] = 0;

    while (!nqueue.empty()) 
    {
        long long y = nqueue.front();
        nqueue.pop();
        nqueue_elements.erase(y);

        auto neighs = adjacencyMap[y];

        for (const auto& n : neighs) 
        {
            if (ndepths.find(n) == ndepths.end()) {
                ndepths[n] = ndepths[y] + 1;

                if (ndepths[n] <= s) {  // Change r to s
                    if (nqueue_elements.find(n) == nqueue_elements.end()) {
                        nqueue.push(n);
                        nqueue_elements.insert(n);
                    }
                }
            }
        }
    }

    // Extract nodes at exactly distance s
    std::vector<int> nodes_at_s;
    for (const auto& [node, depth] : ndepths) {
        if (depth == s) {
            nodes_at_s.push_back(node);
        }
    }

    return nodes_at_s;
}


void computeChunkNodesAtDistance(int start, int end, std::unordered_map<int, std::vector<int>>& adjacencyMap, int s, std::unordered_map<int, std::vector<int>>& nodes_at_distance_s) {
    for (long long x = start; x < end; x++) {
        auto nodes_at_s = getNodesAtDistance(adjacencyMap, s, x);
        
        mtx.lock();
        nodes_at_distance_s[x] = nodes_at_s;
        mtx.unlock();
    }
}

std::unordered_map<int, std::vector<int>> getAllNodesAtDistance(std::unordered_map<int, std::vector<int>>& adjacencyMap, int s, long long num_nodes, int numthreads) {
    std::unordered_map<int, std::vector<int>> nodes_at_distance_s;

    std::vector<std::thread> threads;
    long long chunkSize = num_nodes / numthreads;
    long long remainder = num_nodes % numthreads;

    for (int t = 0; t < numthreads; ++t) {
        int extra = 0;
        for(int j = 0; j < t; ++j)
        {
            if (j < remainder)
                extra++;
        }

        long long start = t * chunkSize + extra;
        long long end = start + chunkSize + (t < remainder ? 1 : 0);

        threads.push_back(std::thread(computeChunkNodesAtDistance, start, end, std::ref(adjacencyMap), s, std::ref(nodes_at_distance_s)));
    }

    for (auto& th : threads) {
        th.join();
    }

    return nodes_at_distance_s;
}

double compCausalGraphCF(const std::vector<std::tuple<int, int>>& edgelist, std::unordered_map<int, std::vector<int>>& adjacencyMap, int r, int s, long long num_nodes, int numthreads) 
{
    std::vector<int> ballVolumes = computeBallVolumes(edgelist, r, num_nodes, numthreads);

    // Compute average of ball volumes
    double sumVolumes = 0.0;
    for (int volume : ballVolumes) sumVolumes += volume;
    double avgVolumes = sumVolumes / num_nodes; // <V_r>

    // Compute neighbors at distance s
    std::unordered_map<int, std::vector<int>> neighbors_at_s = getAllNodesAtDistance(adjacencyMap, s, num_nodes, numthreads);

    // Compute average of products V_r(x)*V_r(y)
    double sumProducts = 0.0;
    long long count = 0;
    for (int x = 0; x < num_nodes; ++x) 
    {
        const auto& neighbors = neighbors_at_s[x];
        for (int i = 0; i < neighbors.size(); ++i) 
        {
            int y = neighbors[i];
            sumProducts += ballVolumes[x] * ballVolumes[y];
            count++;
        }
    }

    double avgProducts = sumProducts / count; // <V_r(x)V_r(y)>

    // Compute and return S_r(s)
    return (avgProducts - avgVolumes * avgVolumes) / (avgVolumes * avgVolumes);
}









// 





int main() {

    std::string filename = "collisions_simid2_chckpt199.txt";//"graphvolcolls.txt";// 


    // std::cout << "Collisions read from "<<filename <<std::endl;
    //
    std::vector<std::tuple<int, int, double>> collisions = readCollisions(filename);


    // for (int i = 0; i< collisions.size();i++) std::cout << std::get<0>(collisions[i]) <<", "<<std::get<1>(collisions[i]) <<", "<<std::get<2>(collisions[i]) << std::endl;

    long long num = 64*64; // WARNING!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    
    int numthreads = 64;
    // std::cout << "Using "<<numthreads<<" threads."<<std::endl;


    // std::cout  << "num = " << num <<std::endl;

    std::vector<std::tuple<int, int>> edgelist = genCausalGraph(collisions, num,numthreads);
    // std::cout << "Causal graph constructed." << std::endl;



    writeCausalGraphToFile(edgelist, "cg_simid2_chckpt199.bin");


    //std::string cgfilename = "cg_simid6_chckpt799.bin"; //"cg_graphvoltest.bin";//
    //std::vector<std::tuple<int, int>> edgelist = readCausalGraphFromFile(cgfilename);
    // std::cout << "Causal graph read from read from "<<cgfilename <<std::endl;

    std::unordered_map<int, std::vector<int>> adjacencyMap = createAdjacencyMap(edgelist,collisions.size());

//    for (int r = 2; r< 13; r++)
//    {
//        std::vector<int> ballVolumes = computeBallVolumes(edgelist, r, collisions.size(), numthreads);
//        double mean = 0.0; double stdev = 0.0;
//        for(int i = 0;i<ballVolumes.size();i++)
//        {
//            mean += ballVolumes[i];
//            stdev+= ballVolumes[i]*ballVolumes[i];
//        }
//        mean/=ballVolumes.size();
//        stdev/=ballVolumes.size();
//        stdev-=mean*mean;
//        stdev = std::sqrt(stdev);
//        std::cout << r <<"\t" <<mean<<"\t"<<stdev<<std::endl;
//
    //}
    int r = 5;
    for(int s = 0;s<20;s++)
    {
         double cgcf = compCausalGraphCF(edgelist, adjacencyMap, r, s, collisions.size(), numthreads);
         std::cout << "r "<<r<<", s "<<s <<", CGCF " <<cgcf <<std::endl;
     }

  

    return 0;
}

// bool areIdentical = true;
//     for (int i = 0; i < edgelist.size(); i++) {
//     int from, to;
//     std::tie(from, to) = edgelist[i];
//     int fromcheck, tocheck;
//     std::tie(fromcheck, tocheck) = edgelistcheck[i];

//     if (from != fromcheck || to != tocheck) {
//         areIdentical = false;
//         break;
//     }
//     }

//     // Print the result.
//     if (areIdentical) {
//     std::cout << "The two vectors are identical." << std::endl;
//     } else {
//     std::cout << "The two vectors are not identical." << std::endl;
//     }


// std::vector<long long> computeBallVolumes(const std::vector<std::tuple<int, int>>& edgelist, long long r, long long num_nodes) {
//     std::vector<long long> ballvolumes(num_nodes);

//     std::unordered_map<int, std::vector<int>> adjacencyMap = createAdjacencyMap(edgelist,num_nodes);
//     // for(int i = 0;i<num_nodes;i++) std::cout << (adjacencyMap[i].size()>2) << std::endl;
//     // std::vector<int>& neighbors0 = adjacencyMap[7];
//     // for (const auto& neighbor : neighbors0) std::cout << neighbor << " ";
//     // std::cout << std::endl;



//     for (long long x = 0; x < num_nodes; x++) {

//         std::queue<long long> nqueue;
//         std::unordered_map<long long, long long> ndepths;
//         std::unordered_set<long long> nqueue_elements;

//         nqueue.push(x);
//         nqueue_elements.insert(x);
//         ndepths[x] = 0;
        
//         long long ballvol = 1;
//         while (!nqueue.empty()) {
//             long long y = nqueue.front();
//             nqueue.pop();
//             nqueue_elements.erase(y);
            
//             auto neighs = adjacencyMap[y];
           
//             for (const auto& n : neighs) 
//             {

//                 if (ndepths.find(n) == ndepths.end()) {
//                     ndepths[n] = ndepths[y] + 1;
                    
//                     if (ndepths[n] <= r) {
//                         ballvol++;
                        
//                         if (nqueue_elements.find(n) == nqueue_elements.end()) {
//                             nqueue.push(n);
//                             nqueue_elements.insert(n);
//                         }
//                     }
//                 }
//             }

//         }
//         ballvolumes[x] = ballvol;
//     }
//     return ballvolumes;
// }

// std::unordered_map<int, std::vector<int>> getAllNodesAtDistance(std::unordered_map<int, std::vector<int>>& adjacencyMap, int s, long long num_nodes) {
//     std::unordered_map<int, std::vector<int>> nodes_at_distance_s;

//     // Iterate over all nodes
//     for (long long x = 0; x < num_nodes; x++) {
//         // Compute nodes at distance s from x and store in the map
//         nodes_at_distance_s[x] = getNodesAtDistance(adjacencyMap, s, x);
//     }

//     return nodes_at_distance_s;
// }

  // std::unordered_map<int, std::vector<int>> neighs_s = getAllNodesAtDistance(adjacencyMap, 4, collisions.size(),numthreads);

    // for(int i = 0;i<collisions.size();i++) std::cout << i <<"\t"<< neighs_s[i].size() <<std::endl;
    // std::vector<int> neigh_s_0 = getNodesAtDistance(adjacencyMap, 4, 141);

    // std::vector<int> neigh_s_1 = getNodesAtDistance(adjacencyMap, 4, 325);

    // std::vector<int> neigh_s_2 = getNodesAtDistance(adjacencyMap, 4, 236);
    // std::cout << "\n" <<std::endl;
    // for(int i = 0;i<neighs_s[141].size();i++) std::cout << std::get<0>(collisions[neighs_s[141][i]]) << ", " <<std::get<1>(collisions[neighs_s[141][i]]) << ", " <<std::get<2>(collisions[neighs_s[141][i]])<< std::endl;
    // std::cout << "\n" <<std::endl;
    // for(int i = 0;i<neighs_s[325].size();i++) std::cout << std::get<0>(collisions[neighs_s[325][i]]) << ", " <<std::get<1>(collisions[neighs_s[325][i]]) << ", " <<std::get<2>(collisions[neighs_s[325][i]])<< std::endl;
    // std::cout << "\n" <<std::endl;
    // for(int i = 0;i<neighs_s[236].size();i++) std::cout << std::get<0>(collisions[neighs_s[236][i]]) << ", " <<std::get<1>(collisions[neighs_s[236][i]]) << ", " <<std::get<2>(collisions[neighs_s[236][i]])<< std::endl;
    




    // std::vector<double> avg;
    // double temp;
    // std::vector<int> Vrx;
    // // std::cout << edgelist.size() << std::endl;
    
    // for(int r = 0; r < 20;r++)
    // {
    //     std::cout << "Computing <V_"<<r<<">" << std::endl;
    //     temp = 0.0;
    //     Vrx = computeBallVolumes(edgelist, r, collisions.size(),numthreads);
    //     for (int i = 0;i<collisions.size();i++) temp+=Vrx[i];
    //     temp/=collisions.size();
    //     avg.push_back(temp);
    // }

    // for(int i = 0; i< avg.size();i++) std::cout << avg[i] << std::endl;

    
    // for (int i = 0;i<collisions.size();i++) std::cout << std::get<0>(collisions[i]) << " " <<std::get<1>(collisions[i]) << " " <<std::get<2>(collisions[i]) << "\t" << Vrx[i]<<std::endl;

    // for(int i= 0;i<edgelist.size();i++) std::cout << std::get<0>(edgelist[i]) << "\t" << std::get<1>(edgelist[i])<<std::endl;

