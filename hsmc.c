/* To compile, use "cc hsmc.c -o sim -lm" */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>

/* Function prototype */
double find_min(double* numbers, int length);

int main(int argc, char *argv[])
{

    int Rind = atoi(argv[1]);
    int it = atoi(argv[2]);
    int totit = it + 25*Rind; 
    printf("Rind = %d, it = %d, totit = %d\n",Rind,it,totit);
    FILE *x_file ; FILE *y_file; FILE *p_file; FILE *vx_file; FILE *vy_file;
    FILE *x_write_file; FILE *y_write_file; 


    char filename[100];
    sprintf(filename,"x_%d.bin",totit); x_file = fopen(filename,"rb"); 
    sprintf(filename,"y_%d.bin",totit); y_file = fopen(filename,"rb"); 
    p_file = fopen("particles.bin","rb"); 
    vx_file = fopen("vx.bin","rb"); 
    vy_file = fopen("vy.bin","rb");
    
    // Simulation Parameters
    int N; N = 500;
    int numsteps = 10000;
    long int numvs = 5000*((long int) numsteps);
    double Rs[31] = {1., 1.005, 1.01, 1.015, 1.02, 1.025, 1.03, 1.035, 1.04, 1.045, 1.05, \
1.055, 1.06, 1.065, 1.07, 1.075, 1.08, 1.085, 1.09, 1.095, 1.1, \
1.105, 1.11, 1.115, 1.12, 1.125, 1.13, 1.135, 1.14, 1.145, 1.15};
    double R = Rs[Rind];
    double x[N];
    double y[N];
    int *particle; particle = malloc(numsteps*sizeof(int));
    double *vx; vx = malloc(numvs*sizeof(double));
    double *vy; vy = malloc(numvs*sizeof(double));
    double dt = 0.05;
    double bs = 50.0;

    fread(x , sizeof(double), N , x_file);
    fread(y , sizeof(double), N , y_file);
    fread(particle , sizeof(long int) , numsteps, p_file);
    fread(vx, sizeof(double) , numvs, vx_file);
    fread(vy, sizeof(double) , numvs, vy_file);
      
    int  p, j; long int vindex;long int i;
    int numtrials = 0; int maxtrials = 1000;
    int numrejects = 0;
    i = 0; vindex = 0; p = 0;
    double xtemp, ytemp, xtrial, ytrial; 
    xtemp = 0.0; ytemp = 0.0; xtrial = 0.0; ytrial = 0.0;
    double sqdists[N-1]; for (i=0;i<N-1;i++) sqdists[i] = 1e15;
    int accept;
    double min; min = 0.0;
    
    // Begin Monte Carlo integration
    ////////////////////////////////
    i=0;
    while(i<numsteps) // Matt : replace 2 with "numsteps"
    {
        if (i%500==0) printf("i = %ld\n",i);
        accept = 0;
        p = particle[i];
        //printf("(x[%d],y[%d]) = (%.17g,%.17g)\n",p,p,x[p],y[p]);
        numtrials = 0;
        while(accept == 0 && numtrials<maxtrials)
        {
            if (vindex<numvs)
            {
                xtrial = x[p] + dt*vx[vindex];
                ytrial = y[p] + dt*vy[vindex];
            }
            else assert(1 == 2);        
            for(j=0;j<N;j++)
            {
                if(j!=p)
                {
            	xtemp = xtrial-x[j];ytemp = ytrial -y[j];
            	if (xtemp > 0.5*bs) xtemp-=bs;
            	if (xtemp < -0.5*bs) xtemp+=bs;

            	if (ytemp > 0.5*bs) ytemp-=bs;
            	if (ytemp < -0.5*bs) ytemp+=bs;

            	sqdists[j] = pow(xtemp,2) + pow(ytemp,2);
                    //printf("dist to particle %d is %.17g\n",j,sqdists[j]);
                }
            }

            min = find_min(sqdists,N-1);


            //printf("min = %.17g\n",min);
            if (min > 4*R*R)
            {
                accept = 1;
                x[p] = xtrial; y[p] = ytrial;
                if (x[p] < -0.5*bs) x[p] += bs;
                if (y[p] < -0.5*bs) y[p] += bs;
                if (x[p] > 0.5*bs) x[p] -= bs;
                if (y[p] > 0.5*bs) y[p] -= bs;
            }
            vindex+=1; // Is this implemented correctly? Shouldn't it update every time? (update, removed "else")
            numtrials+=1;
            if (numtrials >= maxtrials) numrejects+=1;
        }
        i+=1;
    }
    printf("After loop, vindex = %ld and numrejects = %d\n",vindex,numrejects); 

    sprintf(filename,"x_%d.bin",totit+1); x_write_file = fopen(filename,"wb");
    sprintf(filename,"y_%d.bin",totit+1); y_write_file = fopen(filename,"wb");
    fwrite(x,sizeof(double),N,x_write_file);
    fwrite(y,sizeof(double),N,y_write_file);
    fclose(x_file);fclose(y_file); fclose(p_file); fclose(vx_file);fclose(vy_file);
    fclose(x_write_file);fclose(y_write_file);

    free(vx); free(vy); free(particle);
    //fclose(write_file);
    return 0;
}

/* Function definition */
double find_min(double* numbers, int length) {
    /* Initialize the minimum value to the first element in the array */
    double min = numbers[0];
    //printf("    Min function: j = 0, min = %.17g\n",min);
    /* Iterate over the rest of the array and update the minimum value if necessary */
    for (int j = 1; j < length; j++) 
    {
	if (numbers[j] < min) min = numbers[j];
        //printf("    Min function: j = %d, min = %.17g\n",j,min);
    }

    /* Return the minimum value */
    return min;
}
