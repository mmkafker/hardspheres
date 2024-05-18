import numpy as np
import time
import sys
from sortedcontainers import SortedSet
from numba import njit
from matplotlib import pyplot as plt
import matplotlib.patches as patches
from edmd3d_functions import *

#### Create non-PBC time series for each particle ####
def genMSD(AS):
    # Time series will be stored in this variable
    posTS = []
    # Start with initial positions
    posTS.append(AS[0,:,1:3])
    # Record times as well
    tsarr = [0.0]


    for step in range(steps-1):

        # Velocities from previous step
        velsT = AS[step,:,3:5]

        # Positions from previous step
        posT = posTS[-1]

        # Time from previous step
        tT = AS[step,0,0]

        # Time from next step
        tTn = AS[step+1,0,0]

        # Record time
        tsarr.append(tTn)

        # dt
        dtT = tTn-tT

        # Sanity checks
        assert dtT>0
        assert np.max(np.abs(AS[step,:,0]-tT))<1e-14

        # Advance positions at previous step by v*dt
        posTn = posT + dtT*velsT

        # Record positions
        posTS.append(posTn)
    posTS = np.array(posTS)
    tsarr = np.array(tsarr)
    
    #### Compute MSD ####

    # Range of time values at which MSD is to be evaluated
    taus = 10**np.linspace(-5,np.log10(np.max(tsarr)),num=50)
    # taus = np.arange(0,np.min([10,np.max(tsarr)]),0.01)

    # Used to store sum of squared displacements as a function of tau.
    # Each entry refers to a specific tau, summed over time t and particle i.
    all_sds_vs_tau = np.zeros(len(taus))
    # Used to turn the above sum into an average, accounting for the fact
    # that at late times, t+tau is too large, so those data are excluded.
    contains_data = np.zeros(len(taus))

    sdstot = np.zeros(len(taus))
    denom = np.zeros(len(taus))
    # Loop over particles and time
    for i in range(num):
        print(f"Computing MSD for particle {i+1} of {num}.")

        tptaus = np.array([t+taus for t in tsarr])
        itsx = np.interp(tptaus,tsarr,posTS[:,i,0])
        itsy = np.interp(tptaus,tsarr,posTS[:,i,1])

        xts = np.array([posTS[:,i,0] for tau in taus]).T
        yts = np.array([posTS[:,i,1] for tau in taus]).T

        sds0 = (itsx-xts)**2 + (itsy-yts)**2
        tallow = tptaus<=np.max(tsarr)
        sdsi = tallow*sds0
        sdstot += np.sum(sdsi,axis=0)
        denom += np.sum(tallow,axis=0)
    msdstau = sdstot/denom

    stdtau = np.sqrt(sdstot**2/denom - msdstau**2)


    logmsdstau = [( np.log10(msdstau[i+1])-np.log10(msdstau[i]) )/( np.log10(taus[i+1])-np.log10(taus[i]) )
                for i in range(len(taus)-1)]


    plt.figure()
    # plt.errorbar(taus,msdstau,yerr=stdtau,fmt=".-")
    plt.plot(taus,msdstau,".-")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("t")
    plt.ylabel("$\Delta$(t)")

    plt.figure()
    plt.plot(taus[:-1],logmsdstau,".-")
    plt.xscale("log")
    # plt.yscale("log")
    plt.xlabel("t")
    plt.ylabel("Slope")
    plt.show()

@njit
def checkallpairwisedists(posf,num,L):
    md = 1e15
    for i in range(num):
        # print(i)
        for j in range(num):
            if i != j:
                sd = PBCvec3d(posf[i]-posf[j],L)
                sd = np.linalg.norm(sd)

                if sd < md:
                    md = sd
    return md

def checkpairwisedists(posf,num):
    povl = posf @ posf.T
    pwds = np.zeros(int(num*(num-1)/2))
    ind = -1
    for i in range(num-1):
        for j in range(i+1,num):
            ind+=1
            pwds[ind] = povl[i,i]-2*povl[i,j]+povl[j,j]
    return pwds
