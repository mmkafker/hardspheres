#include "edmd_functions.hpp"

#include <cstdio> // for std::remove

void writePositions(const std::vector<std::vector<double>>& state, int simid, int s) {
    // Create the filename
    std::string filename = "pos_" + std::to_string(simid) + ".bin";

    // If s == 0, remove the existing file
    if (s == 0) {
        std::remove(filename.c_str());
    }

    // Open the file in binary mode and append mode
    std::ofstream out(filename, std::ios::binary | std::ios::app);
    if (!out.is_open()) {
        throw std::runtime_error("Could not open file for writing");
    }

    for (const auto& row : state) {
        double x = row[1];
        double y = row[2];
        out.write(reinterpret_cast<const char*>(&x), sizeof(double));
        out.write(reinterpret_cast<const char*>(&y), sizeof(double));
    }

    out.close();
}



void PBCvec(std::vector<double>& x, double L) 
{
    if(std::abs(x[0]) >= 0.5 * L) {
        x[0] -= L * (x[0] > 0 ? 1 : -1);
    }
    if(std::abs(x[1]) >= 0.5 * L) {
        x[1] -= L * (x[1] > 0 ? 1 : -1);
    }
}

std::vector<std::vector<double>> printstate(std::vector<std::vector<double>> state, double L) {
    std::vector<std::vector<double>> statetf = state;
    std::vector<double> times;
    for(auto& st : state)
        times.push_back(st[0]);
    double tmax = *std::max_element(times.begin(), times.end());
    for (int p = 0; p < state.size(); p++) {
        double x = state[p][1];
        double y = state[p][2];
        double vx = state[p][3];
        double vy = state[p][4];
        x += vx * (tmax - times[p]);
        y += vy * (tmax - times[p]);
        std::vector<double> xy = {x,y};
        PBCvec(xy, L);
        statetf[p] = {tmax, x, y, vx, vy};
    }
    return statetf;
}

bool rightcellQ(const std::vector<double>& pos, double xcen, double ycen, double cellsize) {
    return (pos[0] <= xcen + 0.5 * cellsize && pos[0] >= xcen - 0.5 * cellsize) &&
           (pos[1] <= ycen + 0.5 * cellsize && pos[1] >= ycen - 0.5 * cellsize);
}

int cmod(int a, int N) // Because C++ handles mods of negative numbers weirdly.
{
    return ((a % N) + N) % N;
}

std::vector<std::tuple<double, int, int, int, int, int, int>> FindCellCrossings(int particle, std::vector<double>& statep, 
    std::vector<int>& cellind, double xcen, double ycen, double cellsize, int ncells) 
{
    std::vector<std::tuple<double, int, int, int, int, int, int>> events;

    // bool test =rightcellQ({statep[1],statep[2]}, xcen, ycen,  cellsize);
    // if (test == false) 
    // {
    //     std::cout << "Warning: Cell assignment error. ";
    //     std::cout << "Cellsize = "<<cellsize;
    //     std::cout << ". Position: " << statep[1] << " " << statep[2];
    //     std::cout << ".  Cell: [" << xcen-0.5*cellsize << ", " << xcen+0.5*cellsize<<"] x ["<<ycen-0.5*cellsize << ", "<<ycen+0.5*cellsize << "]" << std::endl;
    // }

    if(statep[3] != 0) {
        double tx = ((xcen + (statep[3] > 0 ? 1 : -1) * 0.5 * cellsize - statep[1]) / statep[3]) * (1 + 1e-10);
        if (tx > 0) events.emplace_back(tx + statep[0], 0, particle, cellind[0], cellind[1], 
                            cmod(cellind[0] + (statep[3] > 0 ? 1 : -1),ncells), cmod(cellind[1], ncells));
    }
    if(statep[4] != 0) {
        double ty = ((ycen + (statep[4] > 0 ? 1 : -1) * 0.5 * cellsize - statep[2]) / statep[4]) * (1 + 1e-10);
        if (ty > 0) events.emplace_back(ty + statep[0], 0, particle, cellind[0], cellind[1], 
                            cmod(cellind[0], ncells), cmod(cellind[1] + (statep[4] > 0 ? 1 : -1), ncells));
    }

    return events;
}

std::vector<std::vector<int>> AssignCells(const std::vector<double>& cellcenters, const std::vector<std::vector<double>>& positions, int num) {
    std::vector<std::vector<int>> particle2cell(num, std::vector<int>(2));
    
    for (int i = 0; i < num; i++) {
        double min_distance_x = std::numeric_limits<double>::max();
        double min_distance_y = std::numeric_limits<double>::max();
        for (int j = 0; j < cellcenters.size(); j++) { //does j need to be an int?
            double distance_x = std::abs(positions[i][0] - cellcenters[j]);
            double distance_y = std::abs(positions[i][1] - cellcenters[j]);
            if (distance_x < min_distance_x) {
                min_distance_x = distance_x;
                particle2cell[i][0] = j;
            }
            if (distance_y < min_distance_y) {
                min_distance_y = distance_y;
                particle2cell[i][1] = j;
            }
        }
    }
    
    return particle2cell;
}



std::optional<double> GetCollisionTimeDiff(const std::vector<double>& statei,  const std::vector<double>& statej, double R, double L) {
    double tmax = std::max(statei[0], statej[0]);
    std::vector<double> ri = {statei[1] + (tmax - statei[0]) * statei[3],  statei[2] + (tmax - statei[0]) * statei[4]};
    std::vector<double> rj = {statej[1] + (tmax - statej[0]) * statej[3],  statej[2] + (tmax - statej[0]) * statej[4]};
    std::vector<double> rij = {rj[0]-ri[0],rj[1]-ri[1]}; PBCvec(rij, L);
    std::vector<double> vij = {statej[3] - statei[3], statej[4] - statei[4]};
    double rijsq = rij[0]*rij[0] + rij[1]*rij[1];  
    double vijsq = vij[0]*vij[0] + vij[1]*vij[1];
    double b = rij[0]*vij[0] + rij[1]*vij[1];
    double bsq = b*b;
    double descr = bsq - vijsq * (rijsq - 4.0 * R * R);
    
    if (b < 0 && descr >= 0) {
        return tmax + ( -b - std::sqrt(descr) ) / vijsq;
    }

    return std::nullopt;
}
// USAGE
/////////////////////////////////////////////////////////////////////////////
// std::optional<double> result = GetCollisionTimeDiff(statei, statej, R, L);
// if(result) {  // This will be true if the optional contains a value
//     // A value was returned, we can use it.
//     double collisionTime = result.value();
//     // Create and insert the tuple
// } else {
//     // No collision time was returned, handle this case appropriately
// }

// Alternatively, could just return a NaN?
/////////////////////////////////////////////////////////////////////////////


// namespace std {
//     template <>
//     struct hash<pair<int, int>> {
//         size_t operator()(const pair<int, int>& p) const {
//             auto h1 = hash<int>{}(p.first);
//             auto h2 = hash<int>{}(p.second);
//             return h1 ^ h2;
//         }
//     };
// }
std::unordered_map<std::pair<int, int>, std::vector<int>> GenCell2Particle(const std::vector<std::vector<int>>& particle2cell, int num, int ncells) {
    std::unordered_map<std::pair<int, int>, std::vector<int>> cell2particle;

    for (int i = 0; i < ncells; i++) {
        for (int j = 0; j < ncells; j++) {
            cell2particle[{i, j}] = {};
        }
    }

    for (int i = 0; i < num; i++) {
        cell2particle[{particle2cell[i][0], particle2cell[i][1]}].push_back(i);
    }

    return cell2particle;
}

std::vector<std::tuple<double, int, int, int, int, int, int>> FindCollisions(int particle, 
    const std::vector<std::vector<double>>& state, const std::unordered_map<std::pair<int, int>, std::vector<int>>& cell2particle,
    const std::vector<int> &cellind,int ncells,double R,double L) {
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
        if(iter != cell2particle.end()){
            neighbors.insert(neighbors.end(), iter->second.begin(), iter->second.end());
    }
    }


    neighbors.erase(std::remove(neighbors.begin(), neighbors.end(), particle), neighbors.end());

    std::vector<std::tuple<double, int, int, int, int, int, int>> colls;

    for (int i = 0; i < neighbors.size(); i++) 
    {
        std::optional<double> t = GetCollisionTimeDiff(state[particle], state[neighbors[i]], R, L);
        if(t.has_value()){
            colls.emplace_back(t.value(), 1, particle, neighbors[i], -1, -1, -1);
        }
    }

    return colls;
}

std::unordered_map<int, std::vector<std::tuple<double, int, int, int, int, int, int>>> GenEventDictionary(int num, 
    std::vector<std::vector<double>>& state, const std::unordered_map<std::pair<int, int>, std::vector<int>>& cell2particle,
    int ncells,double R,double L,double cellsize, const std::vector<std::vector<int>>& particle2cell, const std::vector<double>& cellcenters) 
{
    std::unordered_map<int, std::vector<std::tuple<double, int, int, int, int, int, int>>> ed;

    for (int p = 0; p < num; p++) {
        std::vector<int> cell = {particle2cell[p][0], particle2cell[p][1]};
        double xcen = cellcenters[cell[0]];
        double ycen = cellcenters[cell[1]];

        std::vector<std::tuple<double, int, int, int, int, int, int>> colls = FindCollisions(p, state, cell2particle, cell, ncells, R, L);
        std::vector<std::tuple<double, int, int, int, int, int, int>> ccross = FindCellCrossings(p, state[p], cell, xcen, ycen, cellsize, ncells);

        colls.insert(colls.end(), ccross.begin(), ccross.end());
        ed[p] = colls;
    }

    return ed;
}


// struct Compare {
//     bool operator() (const std::tuple<double, int, int, int, int, int, int>& lhs, 
//                      const std::tuple<double, int, int, int, int, int, int>& rhs) const {
//         return std::get<0>(lhs) < std::get<0>(rhs);
//     }
// };
//std::set<std::tuple<double, int, int, int, int, int, int>, Compare> mySet;
using Event = std::tuple<double, int, int, int, int, int, int>;
std::set<Event, Compare> GenEventCalendar(std::unordered_map<int, std::vector<Event>>& ed, int num) {
    std::set<Event, Compare> ec;
    for (int p = 0; p < num; p++) {
        for (const auto& event : ed[p]) {
            ec.insert(event);
        }
    }
    return ec;
}

void HandleCellCrossing(std::tuple<double, int, int, int, int, int, int>& ne, 
                     std::unordered_map<int, std::vector<std::tuple<double, int, int, int, int, int, int>>>& ed, 
                     std::set<std::tuple<double, int, int, int, int, int, int>, Compare>& ec, 
                     std::vector<std::vector<double>>& state, 
                     std::vector<std::vector<int>>& particle2cell, 
                     std::unordered_map<std::pair<int, int>, std::vector<int>>& cell2particle, 
                     int ncells, double R, double L, std::vector<double>& cellcenters, 
                     double cellsize) {
    

    // ty + statep[0], 0, particle, cellind[0], cellind[1], cellind[0] % ncells, (cellind[1] + (statep[4] > 0 ? 1 : -1)) % ncells
    int p = std::get<2>(ne);
    particle2cell[p] = {std::get<5>(ne), std::get<6>(ne)};
    // cell2particle[{std::get<3>(ne), std::get<4>(ne)}].erase(p);

    // Erase
    auto &vec = cell2particle[{std::get<3>(ne), std::get<4>(ne)}];
    auto it = std::find(vec.begin(), vec.end(), p);

    if(it != vec.end())
    {
        vec.erase(it);
    }
    // End erase


    cell2particle[{std::get<5>(ne), std::get<6>(ne)}].push_back(p);

    double t0 = state[p][0];
    double vx = state[p][3];
    double vy = state[p][4];
    double tnew = std::get<0>(ne);

    std::vector<double> new_coord = {state[p][1] + (tnew - t0) * vx, state[p][2] + (tnew - t0) * vy}; PBCvec(new_coord, L);
    state[p] = {tnew, new_coord[0], new_coord[1], vx, vy};

    for (auto ev: ed[p]) {
        ec.erase(ev);
    }

    std::vector<Event> colls = FindCollisions(p, state, cell2particle, particle2cell[p], ncells, R, L);
    std::vector<int> cell = particle2cell[p];
    double xcen = cellcenters[cell[0]];
    double ycen = cellcenters[cell[1]];
    std::vector<Event> ccross = FindCellCrossings(p, state[p], particle2cell[p], xcen, ycen, cellsize, ncells);

    std::vector<Event> events = colls;
    events.insert(events.end(), ccross.begin(), ccross.end());
    ed[p] = events;

    for (auto ev: ed[p]) {
        ec.insert(ev);
    }
}

void HandleCollision(std::tuple<double, int, int, int, int, int, int>& ne, 
                     std::unordered_map<int, std::vector<std::tuple<double, int, int, int, int, int, int>>>& ed, 
                     std::set<std::tuple<double, int, int, int, int, int, int>, Compare>& ec, 
                     std::vector<std::vector<double>>& state, 
                     std::vector<std::vector<int>>& particle2cell, 
                     std::unordered_map<std::pair<int, int>, std::vector<int>>& cell2particle, 
                     int ncells, double R, double L, std::vector<double>& cellcenters, 
                     double cellsize, 
                     std::vector< std::tuple<int, int, double> >& collisions, 
                     int& collcounter) 
{

    int pi = std::get<2>(ne);
    int pj = std::get<3>(ne);
    double tnew = std::get<0>(ne);

    double t0i = state[pi][0];
    double vxi = state[pi][3];
    double vyi = state[pi][4];
    std::vector<double> new_coord_i = {state[pi][1] + (tnew - t0i) * vxi, state[pi][2] + (tnew - t0i) * vyi};PBCvec(new_coord_i, L);
    state[pi] = {tnew, new_coord_i[0], new_coord_i[1], vxi, vyi};

    double t0j = state[pj][0];
    double vxj = state[pj][3];
    double vyj = state[pj][4];
    std::vector<double> new_coord_j = {state[pj][1] + (tnew - t0j) * vxj, state[pj][2] + (tnew - t0j) * vyj};PBCvec(new_coord_j, L);
    state[pj] = {tnew, new_coord_j[0], new_coord_j[1], vxj, vyj};

    // collcounter++;
    collisions.push_back( {pi, pj, tnew});
    // collisions[collcounter] = {pi, pj, tnew};

    std::vector<double> rij = {state[pj][1] - state[pi][1], state[pj][2] - state[pi][2]}; PBCvec(rij, L);
    std::vector<double> vij = {state[pj][3] - state[pi][3], state[pj][4] - state[pi][4]};
    std::vector<double> nhat = {rij[0]/sqrt(rij[0] * rij[0] + rij[1] * rij[1]),rij[1]/sqrt(rij[0] * rij[0] + rij[1] * rij[1])};
    double b = rij[0] * vij[0] + rij[1] * vij[1];
    std::vector<double> dv = {(vij[0]*nhat[0]+vij[1]*nhat[1])*nhat[0],(vij[0]*nhat[0]+vij[1]*nhat[1])*nhat[1]};

    state[pi][3] += dv[0];
    state[pi][4] += dv[1];
    state[pj][3] -= dv[0];
    state[pj][4] -= dv[1];

    std::vector<int> particles = {pi, pj};
    for (int p : particles) {
        for (auto ev : ed[p]) {
            ec.erase(ev);
        }

        std::vector<Event> colls = FindCollisions(p, state, cell2particle, particle2cell[p], ncells, R, L);
        std::vector<int> cell = particle2cell[p];
        double xcen = cellcenters[cell[0]];
        double ycen = cellcenters[cell[1]];
        std::vector<Event> ccross = FindCellCrossings(p, state[p], particle2cell[p], xcen, ycen, cellsize, ncells);

        std::vector<Event> events = colls;
        events.insert(events.end(), ccross.begin(), ccross.end());
        ed[p] = events;

        for (auto ev : ed[p]) {
            ec.insert(ev);
        }
    }

}

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
             double cellsize) {
    auto ne = *(ec.begin());
    double tnew = std::get<0>(ne);
    int etype = std::get<1>(ne);

    if (etype == 0 && ((std::get<3>(ne) < 0 || std::get<4>(ne) < 0) || (std::get<5>(ne) < 0 || std::get<6>(ne) < 0)) )
    {
        std::cout << "Negative cell index!!!"<<std::endl;
        std::cout << "Event: " << std::get<0>(ne) << " " <<std::get<1>(ne) <<" " <<std::get<2>(ne) <<" " <<std::get<3>(ne) <<" " <<std::get<4>(ne) <<" " <<std::get<5>(ne) <<" " <<std::get<6>(ne) << std::endl;
        std::cout << "State: " << state[std::get<2>(ne)][0] <<" " <<state[std::get<2>(ne)][1] <<" " <<state[std::get<2>(ne)][2] <<" " <<state[std::get<2>(ne)][3] <<" " <<state[std::get<2>(ne)][4] << std::endl;
        // assert(false);
    }
    
    if (tnew < 0) {
        // Needs to be updated. Less than the simulation time really, since we are no longer scheduling with deltats,
        // but instead global times
        std::cout << "negative event time" << std::endl;
        std::cout << "Event: " << std::get<0>(ne) << " " <<std::get<1>(ne) <<" " <<std::get<2>(ne) <<" " <<std::get<3>(ne) <<" " <<std::get<4>(ne) <<" " <<std::get<5>(ne) <<" " <<std::get<6>(ne) << std::endl;
        std::cout << etype << std::endl;
        if (etype == 0) {
            std::cout << "State: " << state[std::get<2>(ne)][0] <<" " <<state[std::get<2>(ne)][1] <<" " <<state[std::get<2>(ne)][2] <<" " <<state[std::get<2>(ne)][3] <<" " <<state[std::get<2>(ne)][4] << std::endl;
        }

        
        assert(false);
    }
    
    if (etype == 0) {
        HandleCellCrossing(ne, ed, ec, state, particle2cell, cell2particle, ncells, R, L, cellcenters, cellsize);
    }
    else if (etype == 1) {
        HandleCollision(ne, ed, ec, state, particle2cell, cell2particle, ncells, R, L, cellcenters, cellsize, collisions, collcounter);
    }
}





void HardSphereEDMD(std::vector<std::vector<double>> positions, std::vector<std::vector<double>> velocities, double L, double R, int num, long long steps,
    std::vector<std::vector<double>>& state, std::vector<std::tuple<int, int, double>>& collisions,int simid) 
{
    if(positions.size() != num || velocities.size() != num) 
    {
        std::cout << "Check value of 'num'\n";
        std::cout << "positions.size() = "<<positions.size() <<", velocities.size() = "<<velocities.size()<<", num = "<<num <<std::endl;
        throw std::invalid_argument("num does not match the length of positions or velocities");
    }
    std::vector<double> times(steps, 0.0);

    int ncells = static_cast<int>(std::floor(L / (2 * R)));
    double cellsize = L / ncells;
    std::vector<double> cellcenters(ncells);
    for (int i = 0; i < ncells; ++i) {
        cellcenters[i] = -0.5 * L + 0.5 * cellsize + cellsize * i;
        // std::cout << cellcenters[i] << std::endl;
    }

    std::vector<std::vector<int>> particle2cell = AssignCells(cellcenters, positions, num);
    std::unordered_map<std::pair<int, int>, std::vector<int>> cell2particle = GenCell2Particle(particle2cell, num, ncells);

 
    for (int p = 0; p < num; ++p) {
        state[p] = {0.0, positions[p][0], positions[p][1], velocities[p][0], velocities[p][1]};
    }

    std::unordered_map<int, std::vector<std::tuple<double, int, int, int, int, int, int>>> ed =
        GenEventDictionary(num, state, cell2particle, ncells, R, L, cellsize, particle2cell, cellcenters);

    std::set<Event, Compare> ec = GenEventCalendar(ed, num);

    int collcounter = -1;
    for (long long s = 0; s < steps; ++s) 
    {
        if (s % 100000 == 0) {
            std::cout << s << "\n";
            std::vector<std::vector<double>> statetf = printstate(state,L);

            writePositions(statetf, simid,s);
            // writePositions(statetf, fname);

        }
        if (s% 50000000 == 0) 
        {
            writeCollisions(collisions, simid);
            collisions.clear();
        }
        simstep(ed, ec, state, particle2cell, cell2particle, collcounter, collisions, L, R, ncells, cellcenters, cellsize);
        
        
    }

}


// Function to read binary file into a vector of doubles
std::vector<double> readBinaryFile(const std::string& filename) {
    std::ifstream input(filename, std::ios::binary);
    std::vector<double> data;
    double value;

    while (input.read(reinterpret_cast<char*>(&value), sizeof(double))) {
        data.push_back(value);
    }

    return data;
}

// void writeCollisions(const std::vector<std::tuple<int, int, double>>& collisions, int simid) {
//     std::ofstream outfile;
//     std::string filename = "collisions_" + std::to_string(simid) + ".txt";
//     outfile.open(filename);

//     if (outfile.is_open())
//     {
//         for(const auto& collision : collisions)
//         {
//             outfile << std::get<0>(collision) << "\t"
//                     << std::get<1>(collision) << "\t"
//                     << std::get<2>(collision) << std::endl;
//         }
//         outfile.close();
//     }
//     else
//     {
//         std::cout << "Unable to open file";
//     }
// }

void writeCollisions(const std::vector<std::tuple<int, int, double>>& collisions, int simid) {
    std::ofstream outfile;
    std::string filename = "collisions_" + std::to_string(simid) + ".txt";
    outfile.open(filename, std::ios_base::app);

    if (outfile.is_open())
    {
        for(const auto& collision : collisions)
        {
            outfile << std::get<0>(collision) << "\t"
                    << std::get<1>(collision) << "\t"
                    << std::get<2>(collision) << std::endl;
        }
        outfile.close();
    }
    else
    {
        std::cout << "Unable to open file";
    }
}
