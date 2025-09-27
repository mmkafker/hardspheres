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
    L = 34.7
    R = 1
    v = 1#np.sqrt(2*15264.5/(16*16*16))
    Tmax = 50
    print(f"Simulating up to t = {Tmax}.")

    a0 = 2*R/Tmax

    # run = 5
    # dpath = f"/Users/mattkafker/Documents/WPP/HardSphereSimulations/EventsBasedSimulations/Python/runs/anisotropictemp/ChangeDensity/run{run}"
    # dpath = "/Users/mattkafker/Documents/WPP/HardSphereSimulations/EventsBasedSimulations/Python/runs/LSICs/N4096_L31_3d/T50_run1"
    saveAS = True
    savecolls = True
    saveedges = True
    savefinstate = True
    genCG = True


    num = 16*16*16
    print(f"Packing fraction = {(4*np.pi/3)*num/L**3}")


    # Load initial positions from binary file
    istate = np.fromfile("initstate.bin",dtype=np.float64)
    istate = np.reshape(istate,(num,7))
    IPs = istate[:,1:4]
    print(f"Minimum pairwise distance for initial conditions = {checkallpairwisedists(IPs,num,L)}")



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
    # IPs = np.random.uniform(low=-L/2, high = L/2, size=(num,3))
    
    

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
    
    writestep = 1000
    print(f"Writing states every {writestep} steps.")
    ed, ec, particle2cell,cell2particle, state, collisions, AS = HardSphereEDMD3d(IPs, IVs, L, R, num, Tmax,writestep)
    # ed, ec, particle2cell,cell2particle, state, collisions, AS = LS3d(IPs, IVs, L, R, num, Tmax,writestep,a0)


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
