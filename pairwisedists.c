/* Compute all pairwise distances in box with periodic boundary conditions centered on the origin. */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>


int main(int argc, char *argv[])
{

    int totit = atoi(argv[1]);
    FILE *x_file ; FILE *y_file;
    FILE *seps_write_file; 


    char filename[100];
    sprintf(filename,"x_%d.bin",totit); x_file = fopen(filename,"rb"); 
    sprintf(filename,"y_%d.bin",totit); y_file = fopen(filename,"rb"); 
    
    // Simulation Parameters
    int N; N = 500;
    int numseps = N*(N-1)/2;
    double x[N];
    double y[N];
    double bs = 50.0;

    fread(x , sizeof(double), N , x_file);
    fread(y , sizeof(double), N , y_file);
      
    double *seps; seps = malloc(numseps*sizeof(double));

    int i, j, k;
    k = -1;
    double xsep, ysep;
    for (i=0;i<N-1;i++)
    {
        for (j=i+1;j<N;j++)
        {
            k+=1;
            xsep = x[i]-x[j]; ysep = y[i]-y[j];
            if (xsep > 0.5*bs) xsep-=bs;
            if (xsep < -0.5*bs) xsep+=bs;

            if (ysep > 0.5*bs) ysep-=bs;
            if (ysep < -0.5*bs) ysep+=bs;

            seps[k] = sqrt(pow(xsep,2)+pow(ysep,2));

        }
    }
    

    sprintf(filename,"seps_%d.bin",totit); seps_write_file = fopen(filename,"wb");
    fwrite(seps,sizeof(double),numseps,seps_write_file);
    fclose(x_file);fclose(y_file); 
    fclose(seps_write_file);
    return 0;
}

