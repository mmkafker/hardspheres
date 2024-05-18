from edmd3d_functions import *

def GetCollisionTimeDiff3dLS(statei,statej,L,D0,gamma):
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


    A = vijsq - gamma**2
    B = b - gamma*gamma*tmax-D0*gamma
    C = rijsq - (gamma*tmax)**2 - 2*D0*gamma*tmax-D0*D0
    Bsq = B*B

    descr = Bsq - A*C
    # t = tmax + ( -b-descr**(0.5) )/vijsq
    
    if ((B<=0 or A<0) and descr>=0):
        # print(f"A = {A}, B = {B}, C = {C}, t = {(tmax + ( -B-descr**(0.5) )/A)}")
        return (tmax + ( -B-descr**(0.5) )/A)*(1-np.random.rand()*1e-13) # THIS NOISE APPEARS TO BE IMPORTANT
    return None

def GetCollisionTimeDiff3dLS_(statei,statej,R,L,gamma):
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

    A = vijsq - gamma**2
    B = b - gamma*gamma*tmax
    C = rijsq - (gamma*tmax)**2
    Bsq = B*B

    descr = Bsq - A*C
    # t = tmax + ( -b-descr**(0.5) )/vijsq
    
    if ((B<=0 or A<0) and descr>=0):
        # print(f"A = {A}, B = {B}, C = {C}, t = {(tmax + ( -B-descr**(0.5) )/A)}")
        return (tmax + ( -B-descr**(0.5) )/A)*(1+np.random.rand()*1e-13) # THIS NOISE APPEARS TO BE IMPORTANT
    return None

def FindCollisions3dLS(particle, state,cell2particle,cellind,ncells,R,L,D0,gamma):
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
        t = GetCollisionTimeDiff3dLS(state[particle],state[neigh],L,D0,gamma)
        if t is not None:
            # print(f"Time appended to list is {t}")
            colls.append((t,1,particle,neigh,-1,-1,-1,-1,-1))

    return colls

def GenEventDictionary3dLS(num, state ,cell2particle,ncells,R,L,cellsize,particle2cell,cellcenters,D0,gamma):
    ed = {};

    for p in range(num):

        cell = particle2cell[p]
        xcen, ycen, zcen = cellcenters[cell]
        colls = FindCollisions3dLS(p, state ,cell2particle,cell,ncells,R,L,D0,gamma)
        ccross = FindCellCrossings3d(p, state[p], cell,xcen,ycen,zcen, cellsize,ncells)

        events = colls+ccross
        ed[p] = events



    return ed

def simstep3dLS(ed,ec,state,particle2cell,cell2particle,collcounter,collisions,L,R,ncells,cellcenters,cellsize,D0,gamma): # state is (num,5) tensor with data (t,x,y,vx,vy) for each row representing state of each particle at last update
    ne = ec[0]#.pop(0)
    # print("Next event:")
    # print(ne)
    tnew = ne[0]

    # print(f"tnew = {tnew}")
    # print(f"Event = {ne}")
    etype = ne[1]
    if tnew<0: # Needs to be updated. Less than the simulation time really, since we are no longer scheduling with deltats, but
        # instead global times
        print("negative event time")
        print(ne)
        print(etype)
        if etype==0:
            print(state[ne[2]])
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
            print(state[p])
        assert False

    # if etype == 1:


    

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

        colls = FindCollisions3dLS(p, state,cell2particle,particle2cell[p],ncells,R,L,D0,gamma)
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
        dv = (vij[0]*nhat[0]+vij[1]*nhat[1]+vij[2]*nhat[2]   -   gamma)*nhat # this is the change!
        state[pi,4:7]+=dv#b*nhat/(2.0*R) 
        state[pj,4:7]-=dv#b*nhat/(2.0*R)

        for p in [pi,pj]:
            for ev in ed[p]:
                ec.remove(ev)

            colls = FindCollisions3dLS(p, state,cell2particle,particle2cell[p],ncells,R,L,D0,gamma)
            cell = particle2cell[p]
            xcen, ycen, zcen = cellcenters[cell]
            ccross = FindCellCrossings3d(p, state[p], particle2cell[p],xcen,ycen,zcen,cellsize,ncells)

            events = colls+ccross
            ed[p] = events

            for ev in ed[p]:
                ec.add(ev)

    return ed, ec, particle2cell,cell2particle, state, collisions,collcounter, tnew

def LS3d(positions, velocities, L, R, num, Tmax,writestep,D0,gamma):
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

    # print(cellcenters)

    state = np.zeros((num,7))

    collisions = []

    for p in range(num):
        state[p] = [0.0,positions[p,0],positions[p,1],positions[p,2],
                        velocities[p,0],velocities[p,1],velocities[p,2]]
        
    statetf = printstate3d(state,L)
    allstates.append(statetf)


    ed = GenEventDictionary3dLS(num, state ,cell2particle,ncells,R,L,cellsize,particle2cell,cellcenters,D0,gamma)
    ec = GenEventCalendar3d(ed,num)

    collcounter = -1
    s=-1
    while times[-1]<Tmax:

        if len(times)>1 and times[-1]<times[-2]:
            print(f"Time is going backwards. This should never ever happen.")
            assert False

        s+=1
        
        ed, ec, particle2cell,cell2particle, state, collisions,collcounter,tnew = simstep3dLS(ed,ec,state,particle2cell,cell2particle,collcounter,collisions,L,R,ncells,cellcenters,cellsize,D0,gamma)
        times.append(tnew)

        
        if s%writestep==0:

            print(f"STEP {s}, time = {times[-1]}",flush=True)
            statetf = printstate3d(state,L)
            allstates.append(statetf)

            
            print(f"Current packing fraction = {num*(4*np.pi/3)*(0.5*(D0+gamma*tnew))**3/L**3}")
            ecur = np.sum(np.array([0.5*np.linalg.norm(state[i,4:7])**2 for i in range(num)]))
            print(f"Current energy = {ecur}")
            # if np.sqrt(2*ecur/num) > 10:
            #     for i in range(num):
            #         state[i,4:7]/=np.linalg.norm(state[i,4:7])
            #     # state[i,4:7]*=0
            #     print(f"Energy after rescaling = { np.sum(np.array([0.5*np.linalg.norm(state[i,4:7])**2 for i in range(num)]))}")
            #     del ed
            #     del ec
            #     del particle2cell
            #     del cell2particle
                
            #     particle2cell = AssignCells3d(cellcenters,positions ,num)
            #     cell2particle = GenCell2Particle3d(particle2cell,num,ncells)
            #     ed = GenEventDictionary3dLS(num, state ,cell2particle,ncells,R,L,cellsize,particle2cell,cellcenters,a0)
            #     ec = GenEventCalendar3d(ed,num)

            

            


        

    return ed, ec, particle2cell,cell2particle, state, [[round(x[0]),round(x[1]),x[2]] for x in collisions], allstates
