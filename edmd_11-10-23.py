import numpy as np
import time
import sys
from sortedcontainers import SortedSet
from numba import njit
from matplotlib import pyplot as plt
import matplotlib.patches as patches

@njit
def PBCvec(x,L):
    signs = np.sign(x)
    if abs(x[0]) >= 0.5*L:
        x[0] -=L*signs[0]
    if abs(x[1]) >= 0.5*L:
        x[1] -=L*signs[1]    
    return x

def AssignCells(cellcenters,positions,num):
    particle2cell = np.ndarray(shape=(num, 2), dtype=int)
    for i in range(len(positions)):
        particle2cell[i,0] = np.argmin(np.abs(positions[i,0]-cellcenters))
        particle2cell[i,1] = np.argmin(np.abs(positions[i,1]-cellcenters))

    return particle2cell

def rightcellQ(pos,xcen,ycen,cellsize):
    return (pos[0]<=xcen+0.5*cellsize and pos[0]>=xcen-0.5*cellsize) and (pos[1]<=ycen+0.5*cellsize and pos[1]>=ycen-0.5*cellsize)


def FindCellCrossings_(particle, statep, cellind,xcen,ycen,cellsize,ncells):
    events = []

    vsigns = np.sign(statep[3:5])

    if vsigns[0]!=0:
        tx = (( xcen + vsigns[0]*0.5*cellsize - statep[1] )/statep[3])*(1+1e-10)
        events.append(  (tx+statep[0],0, particle, cellind[0], cellind[1], int((cellind[0]+vsigns[0])%ncells), int(cellind[1]%ncells) )  )
    if vsigns[1]!=0:
        ty = (( ycen + vsigns[1]*0.5*cellsize - statep[2] )/statep[4])*(1+1e-10)
        events.append( (ty+statep[0],0, particle, cellind[0], cellind[1], int(cellind[0]%ncells), int( (cellind[1]+vsigns[1])%ncells) ) )

    return events

@njit
def GetCollisionTimeDiff(statei,statej,R,L):
    tmax = statei[0]
    if statej[0]>statei[0]:
        tmax = statej[0]
    ri = statei[1:3]+(tmax-statei[0])*statei[3:5]
    rj = statej[1:3]+(tmax-statej[0])*statej[3:5]
    rij = PBCvec(rj-ri,L)
    vij = statej[3:5]-statei[3:5]
    rijsq = rij[0]*rij[0]+rij[1]*rij[1]  
    vijsq = vij[0]*vij[0] + vij[1]*vij[1]
    b = rij[0]*vij[0] + rij[1]*vij[1]
    bsq = b*b
    descr = bsq - vijsq*(rijsq - 4.0*R*R);
    # t = tmax + ( -b-descr**(0.5) )/vijsq
    
    if (b<0 and descr>=0):
        return tmax + ( -b-descr**(0.5) )/vijsq #t
    return None

def GenCell2Particle(particle2cell,num,ncells):
    cell2particle = {};

    for i in range(ncells):
        for j in range(ncells):
            cell2particle[(i,j)] = []

    for i in range(num):
        cell2particle[tuple(particle2cell[i])].append(i)

    return cell2particle


def FindCollisions_(particle, state,cell2particle,cellind,ncells,R,L):
    c0 = cellind[0]
    c1 = cellind[1]
    c0p1 = (cellind[0]+1)%ncells
    c0m1 = (cellind[0]-1)%ncells
    c1p1 = (cellind[1]+1)%ncells
    c1m1 = (cellind[1]-1)%ncells
    neighinds = [( c0, c1 ),
                 ( c0, c1m1  ),
                 ( c0, c1p1  ),
                 ( c0m1, c1 ),
                 ( c0m1, c1m1  ),
                 ( c0m1, c1p1  ),
                 ( c0p1, c1 ),
                 ( c0p1, c1m1  ),
                 ( c0p1, c1p1  )
                ]
                
    neighbors = []
    for ind in neighinds:
        neighbors = neighbors + cell2particle[ind]
    neighbors.remove(particle)
    
    colls = []
    
    for neigh in neighbors:
        t = GetCollisionTimeDiff(state[particle],state[neigh],R,L)
        if t is not None:
            colls.append((t,1,particle,neigh,-1,-1,-1))

    return colls

def GenEventDictionary(num, state ,cell2particle,ncells,R,L,cellsize,particle2cell,cellcenters):
    ed = {};

    for p in range(num):

        cell = particle2cell[p]
        xcen, ycen = cellcenters[cell]
        colls = FindCollisions_(p, state ,cell2particle,cell,ncells,R,L)
        ccross = FindCellCrossings_(p, state[p], cell,xcen,ycen,cellsize,ncells)

        events = colls+ccross
        ed[p] = events

    return ed

        
def GenEventCalendar(ed,num):

    ec = SortedSet()
    for p in range(num):
        for event in ed[p]:
            ec.add(event)

    return ec

def simstep(ed,ec,state,particle2cell,cell2particle,collcounter,collisions,L,R,ncells,cellcenters,cellsize): # state is (num,5) tensor with data (t,x,y,vx,vy) for each row representing state of each particle at last update
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
        particle2cell[p] = np.array([ne[5],ne[6]])
        cell2particle[(ne[3],ne[4])].remove(p)
        cell2particle[(ne[5],ne[6])].append(p)

        t0 = state[p,0]
        vx = state[p,3]
        vy = state[p,4]
        xnew,ynew = PBCvec(state[p,1:3]+(tnew-t0)*state[p,3:5],L)
        state[p] = np.array([tnew,xnew,ynew,vx,vy])

        for ev in ed[p]:
            ec.remove(ev)

        colls = FindCollisions_(p, state,cell2particle,particle2cell[p],ncells,R,L)
        cell = particle2cell[p]
        xcen, ycen = cellcenters[cell]
        ccross = FindCellCrossings_(p,state[p], particle2cell[p],xcen,ycen,cellsize,ncells)

        events = colls+ccross
        ed[p] = events

        for ev in ed[p]:
            ec.add(ev)


    
    elif etype == 1: # Collision
        # (tcol,1,particle,neigh,-1,-1,-1)
        pi = ne[2]
        pj = ne[3]
        
        t0i = state[pi,0]
        vxi = state[pi,3]
        vyi = state[pi,4]
        xnewi,ynewi = PBCvec(state[pi,1:3]+(tnew-t0i)*state[pi,3:5],L)
        state[pi] = np.array([tnew,xnewi,ynewi,vxi,vyi]) # WHAT IF T0I!=TOJ!!!

        t0j = state[pj,0]
        vxj = state[pj,3]
        vyj = state[pj,4]
        xnewj,ynewj = PBCvec(state[pj,1:3]+(tnew-t0j)*state[pj,3:5],L)
        state[pj] = np.array([tnew,xnewj,ynewj,vxj,vyj])

        # print(f"toi-toj = {t0i-t0j}")

        collcounter += 1
        collisions[collcounter] = [pi,pj,tnew]

        rij = PBCvec(state[pj,1:3]-state[pi,1:3],L)
        vij = state[pj,3:5]-state[pi,3:5]
        nhat = rij/np.linalg.norm(rij)
        b = rij[0]*vij[0] + rij[1]*vij[1]
        dv = (vij[0]*nhat[0]+vij[1]*nhat[1])*nhat
        state[pi,3:5]+=dv#b*nhat/(2.0*R) 
        state[pj,3:5]-=dv#b*nhat/(2.0*R)

        for p in [pi,pj]:
            for ev in ed[p]:
                ec.remove(ev)

            colls = FindCollisions_(p, state,cell2particle,particle2cell[p],ncells,R,L)
            cell = particle2cell[p]
            xcen, ycen = cellcenters[cell]
            ccross = FindCellCrossings_(p, state[p], particle2cell[p],xcen,ycen,cellsize,ncells)

            events = colls+ccross
            ed[p] = events

            for ev in ed[p]:
                ec.add(ev)

    return ed, ec, particle2cell,cell2particle, state, collisions,collcounter

def printstate(state,L):
    statetf = np.copy(state)
    times = state[:,0]
    tmax = np.max(times)
    tdiffs = tmax-times
    for p in range(len(state)):
        x,y,vx,vy = state[p,1:5]
        x+=vx*tdiffs[p]
        y+=vy*tdiffs[p]
        x,y = PBCvec(np.array([x,y]), L)
        statetf[p] = [tmax,x,y,vx,vy]

    return statetf

def disk_plot(pos_list, box_size, R,filename):
    fig, ax = plt.subplots()

    # Draw the rectangle
    rect = patches.Rectangle((-box_size/2, -box_size/2), box_size, box_size, linewidth=1, edgecolor='black', facecolor='white')
    ax.add_patch(rect)

    # Draw the disks
    for pos in pos_list:
        disk = plt.Circle(pos, R, color='gray', alpha=0.7)
        ax.add_patch(disk)

    # Set plot range and aspect ratio
    ax.set_xlim(-box_size/2, box_size/2)
    ax.set_ylim(-box_size/2, box_size/2)
    ax.set_aspect('equal')

    # Remove axis ticks
    ax.set_xticks([])
    ax.set_yticks([])

    # Save the plot as a high-resolution PNG image
    plt.savefig(filename, bbox_inches='tight', dpi=300)
    plt.close()


def HardSphereEDMD(positions, velocities, L, R, num, steps):
    if(len(positions)!=num or len(velocities)!=num):
        print("Check value of 'num'")
        assert False
    times = np.zeros(steps)

    ncells = int(np.floor(L/(2*R)))
    cellsize = L/ncells
    cellcenters = np.array([-0.5*L+0.5*cellsize+cellsize*i for i in range(ncells)])
    particle2cell = AssignCells(cellcenters,positions ,num)
    cell2particle = GenCell2Particle(particle2cell,num,ncells)

    state = np.zeros((num,5))
    # allpositions = np.zeros((steps,num,2))
    # allvelocities = np.zeros((steps,num,2))

    collisions = np.zeros((steps,3))-1

    for p in range(num):
        state[p] = [0.0,positions[p,0],positions[p,1],velocities[p,0],velocities[p,1]]

    # allpositions[0] = positions
    # allvelocities[0] = velocities

    ed = GenEventDictionary(num, state ,cell2particle,ncells,R,L,cellsize,particle2cell,cellcenters)
    ec = GenEventCalendar(ed,num)

    collcounter = -1
    for s in range(steps):
        if s%100==0:
            print(s)
        # print(f"Energy = {0.5*np.sum(state[:,3:5]**2)}")
        ed, ec, particle2cell,cell2particle, state, collisions,collcounter = simstep(ed,ec,state,particle2cell,cell2particle,collcounter,collisions,L,R,ncells,cellcenters,cellsize)
        # statetf = printstate(state,L)
        # disk_plot(statetf[:,1:3], L, R,f"diskplots/diskplot_{s}.png") 
        # if s%100==0:
        #     statetf = printstate(state,L)
        #     flattened_array = np.array(statetf[:,1:3]).flatten()
        #     with open('pos_sequence.bin', 'ab') as f:
        #         flattened_array.tofile(f)
            # flattened_array.tofile(f"pos_{s}.bin")

    # print(state[:,0])

    return ed, ec, particle2cell,cell2particle, state, [[round(x[0]),round(x[1]),x[2]] for x in collisions[:collcounter]]




def main():    
    
    L = 212.15204627603475
    R = 1
    num = 10000
    steps = 10000
    # IPs = []
    # delta = R*2.1
    data = np.fromfile("IPs.bin", dtype=np.float64)
    # Reshape the 1D array into a 2D array with the desired format
    IPs = data.reshape(-1, 2)
    IVs = np.random.uniform(low=-1, high=1, size=(num, 2))
    pi = np.array([0.0,0.0])
    pf = np.copy(pi)
    Ei = 0.0
    Ef = 0.0
    for p in range(num):
        pi+=IVs[p]
        Ei+=0.5*np.linalg.norm(IVs[p])**2
    

    ed, ec, particle2cell,cell2particle, state, collisions = HardSphereEDMD(IPs, IVs, L, R, num, steps)

    statetf = printstate(state,L)
    for p in range(num):
        pf+=statetf[p,3:5]
        Ef+=0.5*np.linalg.norm(statetf[p,3:5])**2

    print(f"Momentum conservation: {np.abs(pi-pf)}")
    print(f"Energy conservation: {np.abs((Ei-Ef)/Ei)}")
    print(f"Final energy = {Ef}")


    # mindist = np.linalg.norm(PBCvec(statetf[0,1:3]-statetf[1,1:3],L))
    # for p1 in range(num-1):
    #     for p2 in range(p1+1,num):
    #         dd = np.linalg.norm(PBCvec(statetf[p1,1:3]-statetf[p2,1:3],L))
    #         if dd<mindist:
    #             mindist = dd
    #             # print(mindist)
    # print(mindist)  
    # flattened_array = np.array(collisions).flatten()
    # flattened_array.tofile(f"colls.bin")  

    # flattened_array = np.array(statetf[:,1:3]).flatten()
    # flattened_array.tofile(f"posf.bin")  

  
            

    # print(collisions)


    # flattened_array = allpos.flatten()
    # flattened_array.tofile(f"flattened_pos_{unique_id}.bin")

    # flattened_array = allvels.flatten()
    # flattened_array.tofile(f"flattened_vel_{unique_id}.bin")

    # flattened_array = collisions.flatten()
    # flattened_array.tofile(f"flattened_colls_{unique_id}.bin")
    # print(np.diff(times))




if __name__ == '__main__':
    main()