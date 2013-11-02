/*
 * In this MPI program, each generated process creates an array of size m
 * which contains randomly generated numbers. Using a hypercube topology,
 * each process shares its own array with its d dimension neighbor. Each
 * process then shares the arrays it has with its d-1 dimension neighbor,
 * continuing until sharing with its first dimension neighbor. Essentially
 * what this does is every process ends up having then entire array of
 * size n = m*p, where p is the number of processors.
 *
 * The implementation ended up having each process passing around the
 * entire array, filling in the blanks as it gets a new array from its
 * partner process.
 *
 * mpicc -g -Wall -ansi -lm -o gather mpi_allgather.c
 */

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <math.h>
#include <time.h>

int *allocate_array(int n, int id, int m);
int *free_array(int *v);
void print_array(int *v, int n);

#define MAXRAND 100
#define MINRAND 1

int main(int argc, char *argv[])
{
    int m; /* size of individual array */
    int n; /* size of total array */
    int p; /* number of processors */
    int id; /* rank of processors */
    int dim; /* dimension of hypercube */

    int i, counter; /* loop variable */
    int partner;

    int *total_array;
    int *msg; /* incoming message */
    MPI_Status status;

    if(argc != 2) {
        printf("**ERROR: Inappropriate arguments**\n");
        printf("Usage: mpirun -np [# procs] ./gather [size local array]\n");
        exit(1); /* exit program with error */
    }

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &p);
    MPI_Comm_rank(MPI_COMM_WORLD, &id);

    m = atoi(argv[1]);
    n = m * p;
    srand((unsigned)time(NULL) + (unsigned)id); /* seeds random w/ respect to time */
    dim = log(p) / log(2); /* divide by log 2 to get log in base 2 */

    /* Memory allocation */
    total_array = allocate_array(n, id, m);
    msg = allocate_array(n, -1, -1);

    if(m < 4)
        for(i = 0; i < p; i++) {
            MPI_Barrier(MPI_COMM_WORLD);
            if(id == i) {
                printf("This is process %d\n", id);
                print_array(total_array, n);
                printf("\n");
            }
        }

    for(counter = pow(2, dim-1); counter > 0; counter /= 2) {
        partner = counter ^ id;

        if(id > partner) {
            MPI_Send(total_array, n, MPI_INT, partner, id, MPI_COMM_WORLD);
            MPI_Recv(msg, n, MPI_INT, partner, partner, MPI_COMM_WORLD, &status);
        } else {
            MPI_Recv(msg, n, MPI_INT, partner, partner, MPI_COMM_WORLD, &status);
            MPI_Send(total_array, n, MPI_INT, partner, id, MPI_COMM_WORLD);
        }

        for(i = 0; i < n; i++)
            if(msg[i]) /* test if value of array is not 0 */
                total_array[i] = msg[i];
    }

    if(m < 4)
        for(i = 0; i < p; i++) {
            MPI_Barrier(MPI_COMM_WORLD);
            if(id == i) {
                printf("This is process %d\n", id);
                print_array(total_array, n);
                printf("\n");
            }
        }

    /* Deallocate memory */
    total_array = free_array(total_array);
    msg = free_array(msg);

    MPI_Finalize();
    exit(0); /* exit program successfully */
} /* end main */

int *allocate_array(int n, int id, int m)
{
    int i, *v; /* v is pointer to the vector */
    int offset = id * m;

    v = (int*) malloc(n * sizeof(int));

    if(v == NULL) {
        printf("**Error in array allocation**\n");
        return (NULL);
    }

    /* initialize array with random values */
    switch( id ) {
        case -1:
            break;
        default:
            for(i = 0; i < n; i++)
                v[i] = 0;
            for(i = 0; i < m; i++)
                v[offset + i] = (int) rand() % (MAXRAND - MINRAND + 1) + MINRAND;
    }

    return (v); /* returns pointer to the vector */
} /* end array allocation */

int *free_array(int *v)
{
    if(v == NULL)
        return (NULL);

    free(v);
    v = NULL;

    return (v); /* returns a pointer to null */
} /* end free array */

void print_array(int *v, int n)
{
    int i;

    for(i = 0; i < n; i++)
        printf("%6d", v[i]);
    printf("\n");
} /* end print array */

