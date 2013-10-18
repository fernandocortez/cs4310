/*
 * This MPI program multiplies two n by n matrices, A and B, each processor
 * producing a row-band of matrix C. P0 will send row-bands of A and column
 * bands of B to all other processors. Once the processors have utilized their
 * columns of B, they will rotate them in a ring fashion until all processors
 * have calculated their correspoding row-bands of C.
 */

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

void matmul(double *A, double *B, double *C, int n, int p);
void distribute_rows(double *matrix, int n, int p, int id);
void shuffle(double *v, int n, int p, int id);
double *allocate_matrix(int n, int random, char m);
double *free_matrix(double *v);
void print_matrix(double *v, int n, char M);

int main(int argc, char *argv[])
{
    int i; /* loop variable */
    int n; /* dimensions of array */
    int p; /* number of processors */
    int id; /* rank of processor */
    int array_size;
    int row_band;
    double elapsed_time;
    double *matrixA;
    double *matrixB;
    double *matrixC;
    double *globalC; /* global matrix C */

    if(argc != 2) {
        printf("**ERROR: Insufficient arguments**\n");
        printf("Usage: mpirun -np [# procs] ./matmul [matrix size]\n");
        exit(1); /* exit program with error */
    } else {
        n = atoi(argv[1]);
    }

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &p);
    MPI_Comm_rank(MPI_COMM_WORLD, &id);

    n = n>p ? n : p; /* ensures at least 1 row per processor */
    array_size = n * n;
    row_band = array_size / p;

    switch( id ) {
        case 0:
            matrixA = allocate_matrix(array_size, 1, 'A');
            matrixB = allocate_matrix(array_size, 1, 'B');
            matrixC = allocate_matrix(array_size, 0, 'C');
            elapsed_time = -MPI_Wtime();
            break;

        default:
            matrixA = allocate_matrix(row_band, -1, 'a');
            matrixB = allocate_matrix(row_band, -1, 'b');
            matrixC = allocate_matrix(array_size, 0, 'c');
    }

    matmul(matrixA, matrixB, matrixC, n, p);
    if(p > 1) {
        globalC = allocate_matrix(array_size, -1, 'G');
        distribute_rows(matrixA, n, p, id);
        distribute_rows(matrixB, n, p, id);
        for(i = 1; i < p; i++) {
            shuffle(matrixB, n, p, id);
            matmul(matrixA, matrixB, matrixC, n, p);
        }
        MPI_Allgather(matrixC, array_size, MPI_DOUBLE, globalC, array_size, MPI_DOUBLE, MPI_COMM_WORLD);
    }

    if(!id) { /* shorthand for saying id == 0 */
        elapsed_time += MPI_Wtime();
        printf("Elapsed time: %3.3f\n", elapsed_time);

        if(n < 7) {
            print_matrix(matrixA, n, 'A');
            print_matrix(matrixB, n, 'B');
            switch( p ) {
                case 1:
                    print_matrix(matrixC, n, 'C');
                    break;
                default:
                    print_matrix(globalC, n, 'C');
            }
        }
    }

    /* Deallocate memory */
    matrixA = free_matrix(matrixA);
    matrixB = free_matrix(matrixB);
    matrixC = free_matrix(matrixC);
    if(p > 1)
        globalC = free_matrix(globalC);

    MPI_Finalize();
    return 0; /* exit program successfully */
} /* end main */

void matmul(double *A, double *B, double *C, int n, int p)
{
    int i, j, k;
    double temp;
    int rows = n/p;

    for(i = 0; i < rows; i++) {
        for(j = 0; j < n; j++) {
            temp = A[i*n + j];
            for(k = 0; k < n; k++)
                C[i*n + k] += temp * B[j*n + k];
        }
    }
} /* end matrix multiplication */

void distribute_rows(double *matrix, int n, int p, int id)
{
    int i, j;
    int rows = n/p;
    int row_band = n * rows;
    double *msg;
    MPI_Status status;

    if(!id) {
        for(i = 1; i < p; i++) {
            msg = allocate_matrix(row_band, -1, 'm');
            for(j = 0; j < row_band; j++)
                msg[j] = matrix[j + row_band * i];
            MPI_Send(msg, row_band, MPI_DOUBLE, i, i, MPI_COMM_WORLD);
            msg = free_matrix(msg);
        }
    } else {
        MPI_Recv(matrix, row_band, MPI_DOUBLE, 0, id, MPI_COMM_WORLD, &status);
    }
} /* end row distribution */

void shuffle(double *v, int n, int p, int id)
{
    int dest = (id + 1) % p;
    int source = (id - 1) % p;
    MPI_Status status;

    MPI_Sendrecv_replace(v, n*n/p, MPI_DOUBLE, dest, dest, source, source, MPI_COMM_WORLD, &status);
} /* end row-band B shuffle */

double *allocate_matrix(int n, int random, char m)
{
    int i;
    double *v; /* v is pointer to the vector */

    v = (double*) malloc(n * sizeof(double));

    if(v == NULL) {
        printf("**Error in matrix %c allocation: insufficent memory**\n", m);
        return (NULL);
    }

    switch( random ) {
        case 0:
            for(i = 0; i < n; i++)
                v[i] = 0.0;
            break;

        case 1:
            for(i = 0; i < n; i++)
                v[i] = 1.0;
            break;
    }

    return (v); /* returns pointer to the vector */
} /* end matrix allocation */

double *free_matrix(double *v)
{
    if(v == NULL)
        return (NULL);
 
    free(v);
    v = NULL;

    return (v); /* returns a null poiter */
} /* end free matrix */

void print_matrix(double *v, int n, char M)
{
    int row, column, start, end;

    printf("matrix%c\n", M);
    for(row = 0; row < n; row++) {
        start = row * n;
        end = (row + 1) * n;
        for(column = start; column < end; column++)
            printf("%10.2f", v[column]);
        printf("\n");
    }
    printf("\n");
} /* end print matrix */

