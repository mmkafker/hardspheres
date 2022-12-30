/* To compile, use "cc hsmc.c -o sim -lm" */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>

/* Function prototype */
int arg_min(double* numbers, int length);

int main(int argc, char *argv[])
{

    FILE *x_file ; FILE *y_file; 
    FILE *vx_file; FILE *vy_file;
    FILE *x_write_file; FILE *y_write_file; 
    FILE *vx_write_file; FILE *vy_write_file;
    FILE *masses_file;

    char filename[100];
    x_file = fopen("x0.bin","rb"); 
    y_file = fopen("y0.bin","rb"); 
    vx_file = fopen("vx0.bin","rb"); 
    vy_file = fopen("vy0.bin","rb");
    masses_file = fopen("masses.bin","rb");
    

    // Simulation Parameters
    int N = 104329;
    int steps = 150;
    double R = 1.0;
    double dt = 0.1;
    double bs = 680.0;

    // Arrays
    double x0[N];
    double y0[N];
    double vx0[N];
    double vy0[N];
    double masses[N];
    double *x; x = malloc(N*steps*sizeof(double));
    double *y; y = malloc(N*steps*sizeof(double));
    double *vx; vx = malloc(N*steps*sizeof(double));
    double *vy; vy = malloc(N*steps*sizeof(double));
    double dispx[N]; 
    double dispy[N]; 
    double sqdists[N]; 


    // Other variables
    long int i,j,t,k;
    double dist, nx, ny, overlap, dvx, dvy;


    // Zero arrays
    for (i=0;i<N*steps;i++)
    {
        x[i]=0.0;
	y[i]=0.0;
	vx[i]=0.0;
	vy[i]=0.0;
    }

    fread(x0 , sizeof(double), N , x_file);
    fread(y0 , sizeof(double), N , y_file);
    fread(vx0, sizeof(double) , N, vx_file);
    fread(vy0, sizeof(double) , N, vy_file);
    fread(masses, sizeof(double) , N , masses_file);

    for (i=0;i<N;i++)
    {
        x[i]=x0[i];
        y[i]=y0[i];
        vx[i]=vx0[i];
        vy[i]=vy0[i];
    }

    /* Begin time loop  */
    for(t=1;t<steps;t++)
    {
        printf("Step %ld/%d\n",t+1,steps);
	/* Preliminary move  */
	for(i=0;i<N;i++)
	{
	    x[t*N+i] = x[(t-1)*N+i] + vx[(t-1)*N+i]*dt;
	    y[t*N+i] = y[(t-1)*N+i] + vy[(t-1)*N+i]*dt;
	    vx[t*N+i] = vx[(t-1)*N+i];
	    vy[t*N+i] = vy[(t-1)*N+i];
	}

	/* Revise positions due to overlap and boundary  */
	for(i=0;i<N;i++)
	{
            /* Find nearest neighbor  */
            for(j=0;j<N;j++)
	    {
                dispx[j] = x[t*N+j]-x[t*N+i];
		dispy[j] = y[t*N+j]-y[t*N+i];
                if (dispx[j] > 0.5*bs) dispx[j]-=bs;
                if (dispx[j] < -0.5*bs) dispx[j]+=bs;


                if (dispy[j] > 0.5*bs) dispy[j]-=bs;
                if (dispy[j] < -0.5*bs) dispy[j]+=bs;

                sqdists[j] = pow(dispx[j],2) + pow(dispy[j],2);
		if(j==i) sqdists[j] = 1e15;
	    }
            k = arg_min(sqdists,N);
	    if(sqdists[k] < 4*pow(R,2))
	    {
		/* Implement collision  */
		dist = sqrt(sqdists[k]);
		nx = dispx[k]/dist;
		ny = dispy[k]/dist;
                overlap = 2*R-dist;
		x[t*N+i] -= nx*overlap/2;
		y[t*N+i] -= ny*overlap/2;
		x[t*N+k] += nx*overlap/2;
		y[t*N+k] += ny*overlap/2;
                dvx = nx * ( (vx[t*N+k]-vx[t*N+i])*nx + ((vy[t*N+k]-vy[t*N+i]))*ny );
		dvy = ny * ( (vx[t*N+k]-vx[t*N+i])*nx + ((vy[t*N+k]-vy[t*N+i]))*ny );
		vx[t*N+i]+=dvx*2*masses[k]/(masses[i]+masses[k]);
		vy[t*N+i]+=dvy*2*masses[k]/(masses[i]+masses[k]);

		vx[t*N+k]-=dvx*2*masses[i]/(masses[i]+masses[k]);
		vy[t*N+k]-=dvy*2*masses[i]/(masses[i]+masses[k]);
	    }
	}

	/* Implement boundary condition  */
        for(i=0;i<N;i++)
	{
            if (x[t*N+i] < -0.5*bs) x[t*N+i] += bs;
            if (y[t*N+i] < -0.5*bs) y[t*N+i] += bs;
            if (x[t*N+i] > 0.5*bs) x[t*N+i] -= bs;
            if (y[t*N+i] > 0.5*bs) y[t*N+i] -= bs;
	}
    }
      
    

    x_write_file = fopen("xsim.bin","wb");
    y_write_file = fopen("ysim.bin","wb");
    vx_write_file = fopen("vxsim.bin","wb");
    vy_write_file = fopen("vysim.bin","wb");
    fwrite(x,sizeof(double),N*steps,x_write_file);
    fwrite(y,sizeof(double),N*steps,y_write_file);
    fwrite(vx,sizeof(double),N*steps,vx_write_file);
    fwrite(vy,sizeof(double),N*steps,vy_write_file);
    fclose(x_file);
    fclose(y_file);  
    fclose(vx_file);
    fclose(vy_file);
    fclose(x_write_file);
    fclose(y_write_file); 
    fclose(masses_file); 
    fclose(vx_write_file);
    fclose(vy_write_file);

    free(x); free(y); free(vx); free(vy); 
    return 0;
}

int arg_min(double* numbers, int length) {
    double min = numbers[0];
    int argmin = 0;
    for (int j = 1; j < length; j++) 
    {
	if (numbers[j] < min) 
	{
	    min = numbers[j];
	    argmin = j;
	}
    }

    return argmin;
}
