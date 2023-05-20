#ifndef EDMD_FUNCTIONS_HPP
#define EDMD_FUNCTIONS_HPP

#include <set>
#include <tuple>
#include <vector>
#include <cmath>
#include <algorithm>
#include <unordered_map>
#include <utility>
#include <iostream>
#include <limits>
#include <fstream>
#include <random>
#include <cassert>
#include <cstdlib>
#include <string>



void writePositions(const std::vector<std::vector<double>>& state, const std::string& filename);

void PBCvec(std::vector<double>& x, double L) ;


std::vector<std::vector<double>> printstate(std::vector<std::vector<double>> state, double L);

bool rightcellQ(const std::vector<double>& pos, double xcen, double ycen, double cellsize);

int cmod(int a, int N); // Because C++ handles mods of negative numbers weirdly.


std::vector<std::tuple<double, int, int, int, int, int, int>> FindCellCrossings(int particle, std::vector<double>& statep, 
    std::vector<int>& cellind, double xcen, double ycen, double cellsize, int ncells) ;


std::vector<std::vector<int>> AssignCells(const std::vector<double>& cellcenters, const std::vector<std::vector<double>>& positions, int num); 



std::optional<double> GetCollisionTimeDiff(const std::vector<double>& statei,  const std::vector<double>& statej, double R, double L) ;



namespace std {
    template <>
    struct hash<pair<int, int>> {
        size_t operator()(const pair<int, int>& p) const {
            auto h1 = hash<int>{}(p.first);
            auto h2 = hash<int>{}(p.second);
            return h1 ^ h2;
        }
    };
}
std::unordered_map<std::pair<int, int>, std::vector<int>> GenCell2Particle(const std::vector<std::vector<int>>& particle2cell, int num, int ncells);

std::vector<std::tuple<double, int, int, int, int, int, int>> FindCollisions(int particle, 
    const std::vector<std::vector<double>>& state, const std::unordered_map<std::pair<int, int>, std::vector<int>>& cell2particle,
    const std::vector<int> &cellind,int ncells,double R,double L);


std::unordered_map<int, std::vector<std::tuple<double, int, int, int, int, int, int>>> GenEventDictionary(int num, 
    std::vector<std::vector<double>>& state, const std::unordered_map<std::pair<int, int>, std::vector<int>>& cell2particle,
    int ncells,double R,double L,double cellsize, const std::vector<std::vector<int>>& particle2cell, const std::vector<double>& cellcenters) ;



struct Compare {
    bool operator() (const std::tuple<double, int, int, int, int, int, int>& lhs, 
                     const std::tuple<double, int, int, int, int, int, int>& rhs) const {
        return std::get<0>(lhs) < std::get<0>(rhs);
    }
};
//std::set<std::tuple<double, int, int, int, int, int, int>, Compare> mySet;
using Event = std::tuple<double, int, int, int, int, int, int>;
std::set<Event, Compare> GenEventCalendar(std::unordered_map<int, std::vector<Event>>& ed, int num);

void HandleCellCrossing(std::tuple<double, int, int, int, int, int, int>& ne, 
                     std::unordered_map<int, std::vector<std::tuple<double, int, int, int, int, int, int>>>& ed, 
                     std::set<std::tuple<double, int, int, int, int, int, int>, Compare>& ec, 
                     std::vector<std::vector<double>>& state, 
                     std::vector<std::vector<int>>& particle2cell, 
                     std::unordered_map<std::pair<int, int>, std::vector<int>>& cell2particle, 
                     int ncells, double R, double L, std::vector<double>& cellcenters, 
                     double cellsize) ;

void HandleCollision(std::tuple<double, int, int, int, int, int, int>& ne, 
                     std::unordered_map<int, std::vector<std::tuple<double, int, int, int, int, int, int>>>& ed, 
                     std::set<std::tuple<double, int, int, int, int, int, int>, Compare>& ec, 
                     std::vector<std::vector<double>>& state, 
                     std::vector<std::vector<int>>& particle2cell, 
                     std::unordered_map<std::pair<int, int>, std::vector<int>>& cell2particle, 
                     int ncells, double R, double L, std::vector<double>& cellcenters, 
                     double cellsize, 
                     std::vector< std::tuple<int, int, double> >& collisions, 
                     int& collcounter) ;


void simstep(std::unordered_map<int, std::vector<std::tuple<double, int, int, int, int, int, int>>>& ed,
             std::set<std::tuple<double, int, int, int, int, int, int>, Compare>& ec,
             std::vector<std::vector<double>>& state,
             std::vector<std::vector<int>>& particle2cell,
             std::unordered_map<std::pair<int, int>, std::vector<int>>& cell2particle,
             int& collcounter,
             std::vector< std::tuple<int,int, double> >& collisions,
             double L,
             double R,
             int ncells,
             std::vector<double> cellcenters,
             double cellsize);





void HardSphereEDMD(std::vector<std::vector<double>> positions, std::vector<std::vector<double>> velocities, double L, double R, int num, int steps,
    std::vector<std::vector<double>>& state, std::vector<std::tuple<int, int, double>>& collisions,int simid) ;


// Function to read binary file into a vector of doubles
std::vector<double> readBinaryFile(const std::string& filename) ;


#endif