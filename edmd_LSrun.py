import numpy as np
import time
import sys
from sortedcontainers import SortedSet
from numba import njit
from matplotlib import pyplot as plt
import matplotlib.patches as patches
from edmd3d_functions import *
from postprocess import *
from causalgraph import *
from LS3d import *




def main():    

    print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
    print("Running hard sphere simulation.")
    print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")

    

    # np.random.seed(12345) # BE SURE TO TURN THIS OFF FOR LS-PROCEDURE!
    L = 30.25
    R = 1.0+1e-5
    v = 1#np.sqrt(2*15363.5/(16*16*16))
    Tmax = 5
    print(f"Simulating up to t = {Tmax}.")

    
    D0 = 0.0
    LSit = 40000
    gamma = 2*R/(LSit*Tmax)
    print(f"gamma = {gamma}")

    irestart = 1
    restartstep = 6700

    # run = 5
    # dpath = f"/Users/mattkafker/Documents/WPP/HardSphereSimulations/EventsBasedSimulations/Python/runs/anisotropictemp/ChangeDensity/run{run}"
    # dpath = "/Users/mattkafker/Documents/WPP/HardSphereSimulations/EventsBasedSimulations/Python/runs/LSICs/N4096_L31_3d/T50_run1"
    saveAS = False
    savecolls = False
    saveedges = False
    savefinstate = True
    genCG = False


    num = 16*16*16
    print(f"Packing fraction = {(4*np.pi/3)*num/L**3}")


    # Load initial positions from binary file
    # istate = np.fromfile("initstate.bin",dtype=np.float64)
    # istate = np.reshape(istate,(num,7))
    # IPs = istate[:,1:4]
    # print(f"Minimum pairwise distance for initial conditions = {checkallpairwisedists(IPs,num,L)}")

    # Generate initial conditions in a regular grid
    # IPs = []; N13 = 16
    # xs = np.linspace(0,L-3*R,num=N13)
    # if np.diff(xs)[0]<2*R:
    #     assert False
    # ys = np.linspace(0,L-3*R,num=N13)
    # zs = np.linspace(0,L-3*R,num=N13)
    # for x in xs:
    #     for y in ys:
    #         for z in zs:
    #             IPs.append([x+1e-10*(2*np.random.rand()-1),y+1e-10*(2*np.random.rand()-1),z+1e-10*(2*np.random.rand()-1)])
    # IPs = np.array(IPs)
    # COM = np.mean(IPs,axis=0)
    # for i in range(len(IPs)):
    #     IPs[i]-=COM
    

    # Random uniform initial conditions for Lubachevsky-Stillinger
    IPs = np.random.uniform(low=-L/2, high = L/2, size=(num,3))
    
    

    print(f"Particle number: {num}")
    print(f"Box size: {L}")

    # Random unit vectors for velocities
    thetas = np.random.uniform(low=0, high=np.pi, size=num)
    phis   = np.random.uniform(low=0, high=2*np.pi, size = num)
    # IVs = 0*IPs; IVs[0] = np.array([1,0,0])/np.sqrt(2)
    IVs = 0*IPs;
    for i in range(num):
        IVs[i] = v * np.array([ np.sin(thetas[i]) * np.cos(phis[i]), np.sin(thetas[i]) * np.sin(phis[i]), np.cos(thetas[i]) ])


    # !!!!!!!!!!!!!!           WARNING          !!!!!!!!!!!!!!
    # Rescale some velocities for anisotropic initial conditions
    # print("Heating central region.")
    # IVs[np.sqrt(IPs[:,0]**2 + IPs[:,1]**2 + IPs[:,2]**2)<=L/4] *= 10


    pi = np.array([0.0,0.0,0.0])
    pf = np.copy(pi)
    Ei = 0.0
    Ef = 0.0
    for p in range(num):
        pi+=IVs[p]
        Ei+=0.5*np.linalg.norm(IVs[p])**2

    print(f"Initial energy = {Ei}")
    
    writestep = 10000
    print(f"Writing states every {writestep} steps.")
    # ed, ec, particle2cell,cell2particle, state, collisions, AS = HardSphereEDMD3d(IPs, IVs, L, R, num, Tmax,writestep)

    if irestart == 1:
        tfinals = []
    else:
        tfinals = [0.0]
    # state = np.zeros((num,7),dtype=np.float64)
    # for p in range(num):
    #     state[p,1:4] = IPs[p]
    #     state[p,4:7] = IVs[p]
    # statetf = printstate3d(state,L)

    for i in range(restartstep,LSit):
        
        


        if i == restartstep:
            istate = np.fromfile(f"finstate_{i}.bin",dtype=np.float64)
            istate = np.reshape(istate,(num,7))
            IPs = istate[:,1:4]
            IVs = istate[:,4:7]
            restartdata = np.loadtxt(f"checkpoint_{i}.txt")
            D0 = restartdata[0]
            tfinals.append(restartdata[1])
            for p in range(num):
                IVs[p]/=np.linalg.norm(IVs[p])

        elif i > restartstep:
            D0 = tfinals[-1]*gamma
            IPs = statetf[:,1:4]
            IVs = statetf[:,4:7]
            for p in range(num):
                IVs[p]/=np.linalg.norm(IVs[p])
        
        print(f"LS call {i+1}/{LSit}. D0 = {D0}. tfinals[-1] = {tfinals[-1]}.")

        
        ed, ec, particle2cell,cell2particle, state, collisions, AS = LS3d(IPs, IVs, L, R, num, Tmax,writestep,D0,gamma)

        statetf = printstate3d(state,L)
        tfinals.append(tfinals[-1]+statetf[0,0])
        print(f"Minimum pair distance = {checkallpairwisedists(statetf[:,1:4],num,L)}. Smallest it should be is {tfinals[-1]*gamma}.")
        if savefinstate and i%50 == 0 and i > restartstep:
            statetf.tofile(f"finstate_{i}.bin")
            np.savetxt(f"checkpoint_{i}.txt",np.array([tfinals[-1]*gamma,tfinals[-1]]))



    if savecolls:
        # collarr = np.array(collisions)
        print("Writing collisions...")
        collarr = np.zeros((len(collisions),3),dtype=np.float64)
        for i in range(len(collisions)):
            collarr[i,0] = collisions[i][0]
            collarr[i,1] = collisions[i][1]
            collarr[i,2] = collisions[i][2]
        print(f"Number of collisions = {len(collisions)}")
        collarr.tofile("colls.bin")
        print("Collisions successfully written.")

    if genCG:
        print("Constructing causal graph...")
        edgelist = genCausalGraph(collisions,num)
        print("Causal graph successfully constructed.")

    if saveedges:
        print(f"Number of edges = {len(edgelist)}")
        el = np.array(edgelist,dtype=np.float64)
        el.tofile("edges.bin")
    
    
    AS = np.array(AS)
    print(AS.shape)

    if saveAS:
        AS.tofile("AS.bin")

    
    
    

    print(f"Final time = {AS[-1,0,0]}")

    statetf = printstate3d(state,L)
    for p in range(num):
        pf+=statetf[p,4:7]
        Ef+=0.5*np.linalg.norm(statetf[p,4:7])**2


    if savefinstate:
        statetf.tofile("finstate.bin")

    print(f"dP for initial and final momenta: {np.abs(pi-pf)}")
    print(f"dE/E for initial and final energies: {np.abs((Ei-Ef)/Ei)}")
    print(f"Final energy = {Ef}")

    print(f"Minimum pair distance = {checkallpairwisedists(statetf[:,1:4],num,L)}")
    # print(statetf[0,0]*a0)





if __name__ == '__main__':
    main()
