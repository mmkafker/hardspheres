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

std::unordered_map<int, std::unordered_set<int>> createExclusiveNextToAdjacencyMap(const std::unordered_map<int, std::vector<int>>& adjacencyMap) {
    std::unordered_map<int, std::unordered_set<int>> nextToAdjacencyMap;

    // For each node in the adjacency map
    for (const auto& item : adjacencyMap) {
        int node = item.first;
        const std::vector<int>& neighbors = item.second;

        // Initialize an empty set to store nodes at a graph distance of 2
        std::unordered_set<int> nextToNeighbors;

        // For each neighbor of the current node
        for (const int& neighbor : neighbors) {

            // For each neighbor of the neighbor (i.e., next-to-neighbor of the current node)
            for (const int& nextToNeighbor : adjacencyMap.at(neighbor)) {
                nextToNeighbors.insert(nextToNeighbor);
            }
        }

        // Remove direct neighbors from the set of next-to-neighbors
        for (const int& neighbor : neighbors) {
            nextToNeighbors.erase(neighbor);
        }

        // Save the set of next-to-neighbors for the current node in the next-to-adjacency map
        nextToAdjacencyMap[node] = nextToNeighbors;
    }

    return nextToAdjacencyMap;
}






std::vector<long long> computeBallVolumes(const std::vector<std::tuple<int, int>>& edgelist, long long r, long long num_nodes) {
    std::vector<long long> ballvolumes(num_nodes);

    std::unordered_map<int, std::vector<int>> adjacencyMap = createAdjacencyMap(edgelist,num_nodes);

    std::vector<int>& neighbors0 = adjacencyMap[0];
    for (const auto& neighbor : neighbors0) std::cout << neighbor << " ";
    std::cout << std::endl;



    for (long long x = 0; x < num_nodes; x++) {

        std::queue<long long> nqueue;
        std::unordered_map<long long, long long> ndepths;
        std::unordered_set<long long> nqueue_elements;

        nqueue.push(x);
        nqueue_elements.insert(x);
        ndepths[x] = 0;
        
        long long ballvol = 1;
        while (!nqueue.empty()) {
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
        ballvolumes[x] = ballvol;
    }
    return ballvolumes;
}





int main() {

    std::string filename = "graphvolcolls.txt"; 
    std::vector<std::tuple<int, int, double>> collisions = readCollisions(filename);
    // for (int i = 0; i< collisions.size();i++) std::cout << std::get<0>(collisions[i]) <<", "<<std::get<1>(collisions[i]) <<", "<<std::get<2>(collisions[i]) << std::endl;

    long long num = 15;
    int numthreads = 3;

    // std::vector<std::tuple<int, int>> edgelist = genCausalGraph(collisions, num,numthreads);
    // std::cout << "Causal graph constructed." << std::endl;

    // writeCausalGraphToFile(edgelist, "cg_graphvoltest.bin");

    std::vector<std::tuple<int, int>> edgelist = readCausalGraphFromFile("cg_graphvoltest.bin");

    
    // std::cout << edgelist.size() << std::endl;
    std::vector<long long> Vrx = computeBallVolumes(edgelist, 5, collisions.size());
    for (int i = 0;i<collisions.size();i++) std::cout << std::get<0>(collisions[i]) << " " <<std::get<1>(collisions[i]) << " " <<std::get<2>(collisions[i]) << "\t" << Vrx[i]<<std::endl;

    // for(int i= 0;i<edgelist.size();i++) std::cout << std::get<0>(edgelist[i]) << "\t" << std::get<1>(edgelist[i])<<std::endl;


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
