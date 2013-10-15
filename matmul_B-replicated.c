/*
 * This MPI program multiplies two n by n matrices, A and B, with each
 * processor used producing a row-band of matrix C. P0 will send row-bands
 * of A and the entire matrix B to the slave processes.
 */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <mpi.h>

#define MAXRAND 10000
#define MINRAND 1

int *calcBreakpoints(int n, int p);
double *allocate_matrix(int n, int random);
int *allocate_array(int n);
double *free_matrix(double *v);
int *free_array(int *v);
void print_matrix(int n, double *v);
void print_array(int n, int *v);

int main(int argc, char *argv[])
{
    int n;
    int p;
    int id;
    int start;
    int end;
    int dest;
    int source;
    double elapsed_time;
    double *matrixA;
    double *matrixB;
    double *matrixC;
    int *breakPoints;
    MPI_Status status;

    switch( argc )
    {
        case 2:
            n = atoi(argv[1]);
            srand((unsigned)time(NULL)); /* seeds random w/ respect to time */
            break;
        default:
            printf("**ERROR: Insufficient arguments **\n");
            printf("Usage: mpirun -np [# procs] ./matmul [matrix size]\n");
            return 1; /* exit program with error */
    }
    
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &p);
    MPI_Comm_rank(MPI_COMM_WORLD, &id);

    n = n>p ? n : p; /* ensures at least 1 row per processor */
    breakPoints = calcBreakpoints(n, p);

    switch( id )
    {
        case 0:
            matrixA = allocate_matrix(n*n, 1);
            matrixB = allocate_matrix(n*n, 1);
            matrixC = allocate_matrix(n*n, 0);
            break;
        default:
            matrixC = allocate_matrix(n*(breakPoints[id+1]-breakPoints[id]), 0);
    }

    elapsed_time = -MPI_Wtime();



    elapsed_time += MPI_Wtime();
    
    if(!id) {
        printf("Elapsed time: %6.10f\n", elapsed_time);

        if(n < 5) {
            printf("matrixA\n");
            print_matrix(n, matrixA);
            printf("\nmatrixB\n");
            print_matrix(n, matrixB);
            printf("\nmatrixC\n");
            print_matrix(n, matrixC);
            printf("\nbreakPoints\n");
            print_array(p+1, breakPoints);
            printf("\n");
        }
    }

    /* Deallocate memory */
    matrixA = free_matrix(matrixA);
    matrixB = free_matrix(matrixB);
    matrixC = free_matrix(matrixC);
    breakPoints = free_array(breakPoints);

    MPI_Finalize();
    return 0; /* exit program successfully */
} /* end main */

int *calcBreakpoints(int n, int p)
{
    int i, *v; /* v is pointer to the vector */
    int remainder = n%p; /* stores excess rows not evenly distributed */

    v = allocate_array(p+1);

    /* row bands are attempted to be distributed evenly */
    for(i = 1; i <= p; i++)
        v[i] += (v[i-1] + n/p);

    /* The number of remaining rows are spread as evenly as possibled amongst
     * the generated processes. This is done by giving the last k processes
     * and extra row band, where k = remainder.
     */
    for(i = p; i > p-n%p; i--)
        v[i] += remainder--;

    return (v);
} /* end calculate breakpoints */

double *allocate_matrix(int n, int random)
{
    int i;
    double *v; /* v is pointer to the vector */

    v = (double*) malloc(n * sizeof(double));

    if(v == NULL) {
        printf("** Error in array allocation: insufficient memory **\n");
        return (NULL);
    }

    switch( random )
    {
        case 0:
            for(i = 0; i < n; i++)
                v[i] = 0.0;
            break;
        case 1:
            for(i = 0; i < n; i++)
                v[i] = 1.0;
                /* v[i] = (rand() % (MAXRAND - MINRAND + 1) + MINRAND)/100; */
            break;
    }

    return (v); /* returns pointer to the vector */
} /* end allocate matrix */

int *allocate_array(int n)
{
    int i, *v; /* v is pointer to the vector */

    v = (int*) malloc(n * sizeof(int));

    if(v == NULL) {
        printf("**Error in array allocation: insufficient memory **\n");
        return(NULL);
    }

    for(i = 0; i < n; i++)
        v[i] = 0;

    return (v);
} /* end allocate array */

double *free_matrix(double *v)
{
    if(v == NULL)
        return (NULL);

    free(v);
    v = NULL;

    return (v); /* returns a null pointer */
} /* end free matrix */

int *free_array(int *v)
{
    if(v == NULL)
        return (NULL);

    free(v);
    v = NULL;

    return (v); /* returns a null pointer */
} /* end free array */

void print_matrix(int n, double *v)
{
    int row, column, start, end;
    for(row = 0; row < n; row++) {
        start = row * n;
        end = (row + 1) * n;
        for(column = start; column < end; column++)
            printf("%5.2f", v[column]);
        printf("\n");
    }
} /* end print matrix */

void print_array(int n, int *v)
{
    int i;
    for(i = 0; i < n; i++)
        printf("%3d", v[i]);
    printf("\n");
} /* end print array */

