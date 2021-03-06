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

void matmul(double *A, double *B, double *C, int *share, int n, int id);
void distribute_rows(double *matrix, int *share, int n, int p, int id);
void collect_rows(double *matrix, int *share, int n, int p, int id);
int *calc_breakpoints(int n, int p);
double *allocate_matrix(int n, int random, char m);
int *allocate_array(int n);
double *free_matrix(double *v);
int *free_array(int *v);
void print_matrix(int n, double *v);
void print_array(int n, int *v);

int main(int argc, char *argv[])
{
    int n; /* dimensions of array */
    int p; /* number of processors */
    int id; /* rank of process */
    int rows;
    double elapsed_time;
    double *matrixA;
    double *matrixB;
    double *matrixC;
    int *breakPoints;

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
    breakPoints = calc_breakpoints(n, p);

    switch( id )
    {
        case 0:
            matrixA = allocate_matrix(n*n, 1, 'A');
            matrixB = allocate_matrix(n*n, 1, 'B');
            matrixC = allocate_matrix(n*n, 0, 'C');
            elapsed_time = -MPI_Wtime();
            break;
        default:
            rows = breakPoints[id+1] - breakPoints[id];
            matrixA = allocate_matrix(n * rows, -1, 'a');
            matrixC = allocate_matrix(n * rows, 0, 'c');
    }

    MPI_Barrier(MPI_COMM_WORLD);

    switch( p )
    {
        case 1:
            matmul(matrixA, matrixB, matrixC, breakPoints, n, id);
            break;
        default:
            MPI_Bcast(matrixB, n*n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
            distribute_rows(matrixA, breakPoints, n, p, id);
            matmul(matrixA, matrixB, matrixC, breakPoints, n, id);
            collect_rows(matrixC, breakPoints, n, p, id);
    }

    if(!id) {
        elapsed_time += MPI_Wtime();
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

void matmul(double *A, double *B, double *C, int *share, int n, int id)
{
    int i, j, k;
    double temp;
    int start = share[id];
    int end = share[id+1];
    int rows = end - start;
    
    for(i = 0; i < rows; i++) {
        for(j = 0; j < n; j++) {
            temp = A[i*n + j];
            for(k = 0; k < n; k++)
                C[i*n + k] += temp * B[j*n + k];
        }
    }
} /* end matrix multiplication */

void distribute_rows(double *matrix, int *share, int n, int p, int id)
{
    int i, j, rows, start, end;
    double *msg;
    MPI_Status status;
    switch( id )
    {
        case 0:
            for(i = 1; i < p; i++) {
                start = share[i];
                end = share[i+1];
                rows = end - start;
                msg = allocate_matrix(n*rows, -1, 'm');
                for(j = 0; j < n; j++)
                    msg[j] = matrix[j + start * n];
                MPI_Send(msg, n*rows, MPI_DOUBLE, i, i, MPI_COMM_WORLD);
                msg = free_matrix(msg);
            }
            break;
        default:
            rows = share[id+1] - share[id];
            MPI_Recv(matrix, n*rows, MPI_DOUBLE, 0, id, MPI_COMM_WORLD, &status);
    }
} /* end row distribution */

void collect_rows(double *matrix, int *share, int n, int p, int id)
{
    int i, j, rows, start, end;
    double *msg;
    MPI_Status status;
    switch( id )
    {
        case 0:
            for(i = 1; i < p; i++) {
                start = share[i];
                end = share[i+1];
                rows = end - start;
                MPI_Recv(msg, n*rows, MPI_DOUBLE, i, i, MPI_COMM_WORLD, &status);
                for(j = 0; j < n; j++)
                    matrix[j + start * n] = msg[j];
                msg = free_matrix(msg);
            }
        default:
            rows = share[id+1] - share[id];
            MPI_Send(matrix, n*rows, MPI_DOUBLE, 0, id, MPI_COMM_WORLD);
    }
} /* end row collection */

int *calc_breakpoints(int n, int p)
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

double *allocate_matrix(int n, int random, char m)
{
    int i;
    double *v; /* v is pointer to the vector */

    v = (double*) malloc(n * sizeof(double));

    if(v == NULL) {
        printf("** Error in matrix %c allocation: insufficient memory **\n", m);
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

