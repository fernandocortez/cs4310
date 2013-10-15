/*
 * This MPI program multiplies two n by n matrices, A and B, with each
 * processor used producing a row-band of matrix C. P0 will send row-bands
 * of A and the entire matrix B to the slave processes.
 */

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#define MAXRAND 10000
#define MINRAND 1

double *allocate_matrix(int n, int random);
double *free_matrix(int *v);
void print_matrix(int n, double *v);

int main(int argc, char *argv[])
{
    int n;
    int p;
    int id;
    int dest;
    int source;
    double elapsed_time;
    double *matrixA;
    double *matrixB;
    double *matrixC;
    MPI_Status status;

    if(argc != 2) {
        printf("**ERROR: Insufficient arguments **\n");
        printf("Usage: mpirun -np [# processors] ./matmul [matrix size]\n");
        return 1; //exit program with error
    } else {
        n = atoi(argv[1]);
    }
    
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &p);
    MPI_Comm_rank(MPI_COMM_WORLD, &id);
    elapsed_time = -MPI_Wtime();

    if(id == 0) {
        matrixA = allocate_matrix(n, 1);
        matrixB = allocate_matrix(n, 1);
        matrixC = allocate_matrix(n, 0);
    }

    //Deallocate memory
    matrixA = free_matrix(matrixA);
    matrixB = free_matrix(matrixB);
    matrixC = free_matrix(matrixC);

    elapsed_time += MPI_Wtime();
    MPI_Finalize();
    return 0; //exit program successfully
} //end main

double *allocate_matrix(int n, int random)
{
    int i, *v; // *v is pointer to the vector
    int size = n * n;

    v = (double*) malloc(size * sizeof(double));

    if(v == NULL) {
        printf("** Error in array allocation: insufficient memory **\n");
        return (NULL);
    }

    switch( random )
    {
        case 0:
            for(i = 0; i < size; i++)
                v[i] = 0.0;
            break;
        case 1:
            for(i = 0; i < size; i++)
                v[i] = 1.0;
                //v[i] = (rand() % (MAXRAND - MINRAND + 1) + MINRAND)/100;
            break;
    }

    return (v); //returns pointer to the vector
} //end allocate matrix

double *free_matrix(int *v)
{
    if(v == NULL)
        return (NULL);

    free(v);
    v = NULL;

    return (v); //returns a null pointer
} //end free matrix

void print_matrix(int n, double *v)
{
    int row, column;
    for(row = 0; row < n; row++) {
        for(column = row * n; column < (row+1)*n; column++)
            printf("%5.2f", v[i]);
        printf("\n");
    }
} //end print matrix

