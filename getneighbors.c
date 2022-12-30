/* 
 * Find all "neighbors" of all spheres in a list, which is defined as all spheres within three radii
 * of the sphere in question.
 *
 * BE SURE YOU CHANGE
 * - R
 * - bs
 * - N
 *
 * */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>


int main(int argc, char *argv[])
{

    
    int Rind = atoi(argv[1]);
    int fileind = atoi(argv[2]);
    FILE *x_file ; FILE *y_file;
    FILE *write_file; 


    char filename[100];
    sprintf(filename,"x_%d.bin",fileind); x_file = fopen(filename,"rb");
    sprintf(filename,"y_%d.bin",fileind); y_file = fopen(filename,"rb"); 
    
    // Simulation Parameters
    int N; N = 500;
    double *x; x = malloc(N*sizeof(double));
    double *y; y = malloc(N*sizeof(double));
    double bs = 50.0;

    double Rs[31] = {1., 1.005, 1.01, 1.015, 1.02, 1.025, 1.03, 1.035, 1.04, 1.045, 1.05, \
1.055, 1.06, 1.065, 1.07, 1.075, 1.08, 1.085, 1.09, 1.095, 1.1, \
1.105, 1.11, 1.115, 1.12, 1.125, 1.13, 1.135, 1.14, 1.145, 1.15};
    double R = Rs[Rind];

    fread(x , sizeof(double), N , x_file);
    fread(y , sizeof(double), N , y_file);
    int maxneigh = 10;
    int M = N*maxneigh;
    int *neighbors; neighbors = malloc(M*sizeof(int));
    int i, j, k;
    double xsep, ysep, sep;
    
    for (i=0;i<M;i++) neighbors[i] = -1;
    
    for (i=0;i<N;i++)
    {
        k = 0; // To keep track of each neighbor we find.
	for(j=0;j<N;j++)
	{
            xsep = x[i]-x[j]; ysep = y[i]-y[j];
            if (xsep > 0.5*bs) xsep-=bs;
            if (xsep < -0.5*bs) xsep+=bs;

            if (ysep > 0.5*bs) ysep-=bs;
            if (ysep < -0.5*bs) ysep+=bs;

            sep = pow(xsep,2)+pow(ysep,2);
	    if (i==j) sep = 1e15;

	    if (sep < 9.0*R*R)
	    {
                //printf("sep = %.17g, k = %d\n",sep,k);
		neighbors[i*maxneigh+k] = j+1; //Mathematica indexing because I'm processing in Mathematica
	        k+=1;
	    }
	}
	//printf("Next particle\n");
    }
    
    sprintf(filename,"neighbors_%d.bin",fileind); write_file = fopen(filename,"wb");
    fwrite(neighbors,sizeof(int),M,write_file);
    fclose(x_file);fclose(y_file); 
    fclose(write_file);
    
    free(neighbors);
    
    return 0;
}

