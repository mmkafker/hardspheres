import numpy as np

num=500;
BoxSize=50;
ParticleRadius=1;
ThermalizeSteps=50;
StepSize=0.1;
PrintData=False;
WritePositions=False;
MCSteps=10000;
alpha = 5000;
mass=1.0;
beta=1.0;

vx = np.random.normal( loc=0.0,scale=1/np.sqrt(mass*beta) , size=alpha*MCSteps)
vxfile = open("vx.bin","wb")
vxfile.write(vx)
vxfile.close()


vy = np.random.normal( loc=0.0,scale=1/np.sqrt(mass*beta) , size=alpha*MCSteps)
vyfile = open("vy.bin","wb")
vyfile.write(vy)
vyfile.close()


particles = np.random.randint(0,num,MCSteps)
pfile = open("particles.bin","wb")
pfile.write(particles.astype("int32"))
pfile.close()
