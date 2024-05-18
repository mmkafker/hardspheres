import numpy as np
import time
import sys
from sortedcontainers import SortedSet
from numba import njit
from matplotlib import pyplot as plt
import matplotlib.patches as patches

@njit
def PBCvec3d(x,L):
    signs = np.sign(x)
    if abs(x[0]) >= 0.5*L:
        x[0] -=L*signs[0]
    if abs(x[1]) >= 0.5*L:
        x[1] -=L*signs[1] 
    if abs(x[2]) >= 0.5*L:
        x[2] -=L*signs[2]    
    return x

def AssignCells3d(cellcenters,positions,num):
    particle2cell = np.ndarray(shape=(num, 3), dtype=int)
    for i in range(len(positions)):
        particle2cell[i,0] = np.argmin(np.abs(positions[i,0]-cellcenters))
        particle2cell[i,1] = np.argmin(np.abs(positions[i,1]-cellcenters))
        particle2cell[i,2] = np.argmin(np.abs(positions[i,2]-cellcenters))

    return particle2cell

def rightcellQ3d(pos,xcen,ycen,zcen,cellsize):
    return (pos[0]<=xcen+0.5*cellsize and pos[0]>=xcen-0.5*cellsize) and (
        pos[1]<=ycen+0.5*cellsize and pos[1]>=ycen-0.5*cellsize) and (
        pos[2]<=zcen+0.5*cellsize and pos[2]>=zcen-0.5*cellsize)

def FindCellCrossings3d(particle, statep, cellind,xcen,ycen,zcen,cellsize,ncells):
    events = []

    vsigns = np.sign(statep[4:7])
    vsigns[0] = int(vsigns[0])
    vsigns[1] = int(vsigns[1])
    vsigns[2] = int(vsigns[2])


    if vsigns[0]!=0:
        tx = (( xcen + vsigns[0]*0.5*cellsize - statep[1] )/statep[4])*(1+np.random.rand()*1e-8)
        # if tx < 0:
            # print(f"xcen = {xcen}\nvsigns[0] = {vsigns[0]}\nvsigns[0]*0.5*cellsize = {vsigns[0]*0.5*cellsize}\nstatep[1] = {statep[1]}\nnum = {( xcen + vsigns[0]*0.5*cellsize - statep[1] )}\ndenom = {statep[4]}\nt without randomenss = {(( xcen + vsigns[0]*0.5*cellsize - statep[1] )/statep[4])}\ntx = {tx}")
        #     assert False
        if tx>0:
            events.append( (tx+statep[0],0, particle, cellind[0], cellind[1], cellind[2], 
                            round((cellind[0]+vsigns[0])%ncells), 
                            cellind[1],  
                            cellind[2]) )
    if vsigns[1]!=0:
        ty = (( ycen + vsigns[1]*0.5*cellsize - statep[2] )/statep[5])*(1+np.random.rand()*1e-8)
        # if ty < 0:
            # print(f"ycen = {ycen}\nvsigns[1] = {vsigns[1]}\nvsigns[1]*0.5*cellsize = {vsigns[1]*0.5*cellsize}\nstatep[2] = {statep[2]}\nnum = {( ycen + vsigns[1]*0.5*cellsize - statep[2] )}\ndenom = {statep[5]}\nt without randomenss = {(( ycen + vsigns[1]*0.5*cellsize - statep[2] )/statep[5])}\nty = {ty}")
        #     assert False
        if ty > 0:
            events.append( (ty+statep[0],0, particle, cellind[0], cellind[1], cellind[2],
                        cellind[0], 
                        round((cellind[1]+vsigns[1])%ncells),
                        cellind[2]) )
    if vsigns[2]!=0:
        tz = (( zcen + vsigns[2]*0.5*cellsize - statep[3] )/statep[6])*(1+np.random.rand()*1e-8)
        # if tz < 0:
            # print(f"zcen = {zcen}\nvsigns[2] = {vsigns[2]}\nvsigns[2]*0.5*cellsize = {vsigns[2]*0.5*cellsize}\nstatep[3] = {statep[3]}\nnum = {( zcen + vsigns[2]*0.5*cellsize - statep[3] )}\ndenom = {statep[6]}\nt without randomenss = {(( zcen + vsigns[2]*0.5*cellsize - statep[3] )/statep[6])}\ntz = {tz}")
        #     assert False
        if tz > 0:
            events.append( (tz+statep[0],0, particle, cellind[0], cellind[1], cellind[2],
                        cellind[0],
                        cellind[1],
                        round((cellind[2]+vsigns[2])%ncells)) )

    return events

@njit
def GetCollisionTimeDiff3d(statei,statej,R,L):
    tmax = statei[0]
    if statej[0]>statei[0]:
        tmax = statej[0]
    ri = statei[1:4]+(tmax-statei[0])*statei[4:7]
    rj = statej[1:4]+(tmax-statej[0])*statej[4:7]
    rij = PBCvec3d(rj-ri,L)
    vij = statej[4:7]-statei[4:7]
    rijsq = rij[0]*rij[0] + rij[1]*rij[1] + rij[2]*rij[2]
    vijsq = vij[0]*vij[0] + vij[1]*vij[1] + vij[2]*vij[2]
    b =     rij[0]*vij[0] + rij[1]*vij[1] + rij[2]*vij[2]
    bsq = b*b
    descr = bsq - vijsq*(rijsq - 4.0*R*R);
    # t = tmax + ( -b-descr**(0.5) )/vijsq
    
    if (b<0 and descr>=0):
        return (tmax + ( -b-descr**(0.5) )/vijsq)*(1+np.random.rand()*1e-13) # THIS NOISE APPEARS TO BE IMPORTANT
    return None


def GenCell2Particle3d(particle2cell,num,ncells):
    cell2particle = {};

    for i in range(ncells):
        for j in range(ncells):
            for k in range(ncells):
                cell2particle[(i,j,k)] = []

    for i in range(num):
        cell2particle[tuple(particle2cell[i])].append(i) # Probably needs to be verified

    return cell2particle

def FindCollisions3d(particle, state,cell2particle,cellind,ncells,R,L):
    c0 = cellind[0]
    c1 = cellind[1]
    c2 = cellind[2]
    c0p1 = (cellind[0]+1)%ncells
    c0m1 = (cellind[0]-1)%ncells
    c1p1 = (cellind[1]+1)%ncells
    c1m1 = (cellind[1]-1)%ncells
    c2p1 = (cellind[2]+1)%ncells
    c2m1 = (cellind[2]-1)%ncells
    neighinds = [( c0  , c1  , c2 ),
                 ( c0  , c1m1, c2 ),
                 ( c0  , c1p1, c2 ),
                 ( c0m1, c1  , c2 ),
                 ( c0m1, c1m1, c2 ),
                 ( c0m1, c1p1, c2 ),
                 ( c0p1, c1  , c2 ),
                 ( c0p1, c1m1, c2 ),
                 ( c0p1, c1p1, c2 ),

                 ( c0  , c1  , c2m1 ),
                 ( c0  , c1m1, c2m1 ),
                 ( c0  , c1p1, c2m1 ),
                 ( c0m1, c1  , c2m1 ),
                 ( c0m1, c1m1, c2m1 ),
                 ( c0m1, c1p1, c2m1 ),
                 ( c0p1, c1  , c2m1 ),
                 ( c0p1, c1m1, c2m1 ),
                 ( c0p1, c1p1, c2m1 ),

                 ( c0  , c1  , c2p1 ),
                 ( c0  , c1m1, c2p1 ),
                 ( c0  , c1p1, c2p1 ),
                 ( c0m1, c1  , c2p1 ),
                 ( c0m1, c1m1, c2p1 ),
                 ( c0m1, c1p1, c2p1 ),
                 ( c0p1, c1  , c2p1 ),
                 ( c0p1, c1m1, c2p1 ),
                 ( c0p1, c1p1, c2p1 )
                ]
                
    neighbors = []
    for ind in neighinds:
        neighbors = neighbors + cell2particle[ind]
    neighbors.remove(particle)
    
    colls = []
    
    for neigh in neighbors:
        t = GetCollisionTimeDiff3d(state[particle],state[neigh],R,L)
        if t is not None:
            colls.append((t,1,particle,neigh,-1,-1,-1,-1,-1))

    return colls

def GenEventDictionary3d(num, state ,cell2particle,ncells,R,L,cellsize,particle2cell,cellcenters):
    ed = {};

    for p in range(num):

        cell = particle2cell[p]
        xcen, ycen, zcen = cellcenters[cell]
        colls = FindCollisions3d(p, state ,cell2particle,cell,ncells,R,L)
        ccross = FindCellCrossings3d(p, state[p], cell,xcen,ycen,zcen, cellsize,ncells)

        events = colls+ccross
        ed[p] = events

    return ed

def GenEventCalendar3d(ed,num):

    ec = SortedSet()
    for p in range(num):
        for event in ed[p]:
            ec.add(event)

    return ec


def simstep3d(ed,ec,state,particle2cell,cell2particle,collcounter,collisions,L,R,ncells,cellcenters,cellsize): # state is (num,5) tensor with data (t,x,y,vx,vy) for each row representing state of each particle at last update
    ne = ec[0]#.pop(0)
    #print(ne)
    tnew = ne[0]
    # print(f"Event = {ne}")
    etype = ne[1]
    if tnew<0: # Needs to be updated. Less than the simulation time really, since we are no longer scheduling with deltats, but
        # instead global times
        print("negative event time")
        print(ne)
        print(etype)
        if etype==0:
            print(state[ne[2]])
        assert False

    

    if etype == 0: # Cell crossing
        # (ty+tlast,0, particle, cellind[0], cellind[1], int(cellind[0]%ncells), int( (cellind[1]+vsigns[1])%ncells) )
        p = ne[2]
        particle2cell[p] = np.array([ne[6],ne[7],ne[8]])
        cell2particle[(ne[3],ne[4],ne[5])].remove(p)
        cell2particle[(ne[6],ne[7],ne[8])].append(p)

        t0 = state[p,0]
        vx = state[p,4]
        vy = state[p,5]
        vz = state[p,6]

        xnew,ynew,znew = PBCvec3d(state[p,1:4]+(tnew-t0)*state[p,4:7],L)
        state[p] = np.array([tnew,xnew,ynew,znew,vx,vy,vz])

        for ev in ed[p]:
            ec.remove(ev)

        colls = FindCollisions3d(p, state,cell2particle,particle2cell[p],ncells,R,L)
        cell = particle2cell[p]
        xcen, ycen, zcen = cellcenters[cell]
        ccross = FindCellCrossings3d(p,state[p], particle2cell[p],xcen,ycen,zcen,cellsize,ncells)

        events = colls+ccross
        ed[p] = events

        for ev in ed[p]:
            ec.add(ev)


    
    elif etype == 1: # Collision
        # (tcol,1,particle,neigh,-1,-1,-1,-1,-1)
        pi = ne[2]
        pj = ne[3]
        
        t0i = state[pi,0]
        vxi = state[pi,4]
        vyi = state[pi,5]
        vzi = state[pi,6]
        xnewi,ynewi,znewi = PBCvec3d(state[pi,1:4]+(tnew-t0i)*state[pi,4:7],L)
        state[pi] = np.array([tnew,xnewi,ynewi,znewi,vxi,vyi,vzi]) # WHAT IF T0I!=TOJ!!!

        t0j = state[pj,0]
        vxj = state[pj,4]
        vyj = state[pj,5]
        vzj = state[pj,6]
        xnewj,ynewj,znewj = PBCvec3d(state[pj,1:4]+(tnew-t0j)*state[pj,4:7],L)
        state[pj] = np.array([tnew,xnewj,ynewj,znewj,vxj,vyj,vzj])

        # print(f"toi-toj = {t0i-t0j}")

        collcounter += 1
        collisions.append([pi,pj,tnew])

        rij = PBCvec3d(state[pj,1:4]-state[pi,1:4],L)
        vij = state[pj,4:7]-state[pi,4:7]
        nhat = rij/np.linalg.norm(rij)
        b = rij[0]*vij[0] + rij[1]*vij[1] + rij[2]*vij[2]
        dv = (vij[0]*nhat[0]+vij[1]*nhat[1]+vij[2]*nhat[2])*nhat
        state[pi,4:7]+=dv#b*nhat/(2.0*R) 
        state[pj,4:7]-=dv#b*nhat/(2.0*R)

        for p in [pi,pj]:
            for ev in ed[p]:
                ec.remove(ev)

            colls = FindCollisions3d(p, state,cell2particle,particle2cell[p],ncells,R,L)
            cell = particle2cell[p]
            xcen, ycen, zcen = cellcenters[cell]
            ccross = FindCellCrossings3d(p, state[p], particle2cell[p],xcen,ycen,zcen,cellsize,ncells)

            events = colls+ccross
            ed[p] = events

            for ev in ed[p]:
                ec.add(ev)

    return ed, ec, particle2cell,cell2particle, state, collisions,collcounter, tnew

def printstate3d(state,L):
    statetf = np.copy(state)
    times = state[:,0]
    tmax = np.max(times)
    tdiffs = tmax-times
    for p in range(len(state)):
        x,y,z,vx,vy,vz = state[p,1:7]
        x+=vx*tdiffs[p]
        y+=vy*tdiffs[p]
        z+=vz*tdiffs[p]
        x,y,z = PBCvec3d(np.array([x,y,z]), L)
        statetf[p] = [tmax,x,y,z,vx,vy,vz]

    return statetf

def HardSphereEDMD3d(positions, velocities, L, R, num, Tmax,writestep):
    if(len(positions)!=num or len(velocities)!=num):
        print("Check value of 'num'")
        assert False
    times = [0]

    allstates = []

    ncells = int(np.floor(L/(2*R)))
    cellsize = L/ncells
    cellcenters = np.array([-0.5*L+0.5*cellsize+cellsize*i for i in range(ncells)])
    particle2cell = AssignCells3d(cellcenters,positions ,num)
    cell2particle = GenCell2Particle3d(particle2cell,num,ncells)

    print(f"Cell size = {cellsize}")

    state = np.zeros((num,7))

    collisions = []

    for p in range(num):
        state[p] = [0.0,positions[p,0],positions[p,1],positions[p,2],
                        velocities[p,0],velocities[p,1],velocities[p,2]]
        
    statetf = printstate3d(state,L)
    allstates.append(statetf)


    ed = GenEventDictionary3d(num, state ,cell2particle,ncells,R,L,cellsize,particle2cell,cellcenters)
    ec = GenEventCalendar3d(ed,num)

    collcounter = -1
    s=-1
    while times[-1]<Tmax:
        s+=1
        
        ed, ec, particle2cell,cell2particle, state, collisions,collcounter,tnew = simstep3d(ed,ec,state,particle2cell,cell2particle,collcounter,collisions,L,R,ncells,cellcenters,cellsize)
        times.append(tnew)

                # print()
        if s%writestep==0:
            print(f"STEP {s}, time = {times[-1]}",flush=True)

            statetf = printstate3d(state,L)
            allstates.append(statetf)


        

    return ed, ec, particle2cell,cell2particle, state, [[round(x[0]),round(x[1]),x[2]] for x in collisions], allstates
