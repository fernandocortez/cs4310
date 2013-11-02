/*
 * In this MPI program, each generated process creates an array of size m
 * which contains randomly generated numbers. This local array is sorted using
 * merge sort. Using a hypercube topology, each process shares its local array
 * with its d dimension neighbor. Every time each processor receives a new piece
 * of the total array it will merge with its local copy of the total array. Each
 * process then shares its local copy of the total array with its d-1 dimension
 * neighbor, recursively repeating until sharing with its first dimension
 * neighbor. Essentially what this does is every process ends up having then
 * entire array of size n = m*p, where p is the number of processors.
 *
 * Merge sort algorithm was found at:
 *  rosettacode.org/wiki/Sorting_algorithms/Merge_sort#C
 *
 * mpicc -g -Wall -ansi -lm -o sort mpi_mergesort.c
 */

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <math.h>
#include <time.h>

void merge(int *list, int left_start, int left_end, int right_start, int right_end);
void mergesort_iter(int left, int right, int *list);
void mergesort(int *list, int length);
int *allocate_array(int n, int random);
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

    int *local_array;
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
    local_array = allocate_array(m, 1);
    total_array = allocate_array(n, 0);
    msg = allocate_array(n, -1);

    mergesort(local_array, m);

    /* Print tests */
    if(m < 4)
        for(i = 0; i < p; i++) {
            MPI_Barrier(MPI_COMM_WORLD);
            if(id == i) {
                printf("This is process %d local array\n", id);
                print_array(local_array, m);
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

    /* Print tests */
    if(m < 4)
        for(i = 0; i < p; i++) {
            MPI_Barrier(MPI_COMM_WORLD);
            if(id == i) {
                printf("This is process %d total array\n", id);
                print_array(total_array, n);
                printf("\n");
            }
        }

    /* Deallocate memory */
    local_array = free_array(local_array);
    total_array = free_array(total_array);
    msg = free_array(msg);

    MPI_Finalize();
    exit(0); /* exit program successfully */
} /* end main */

void merge(int *list, int left_start, int left_end, int right_start, int right_end)
{
    /* calculate temporary list sizes */
    int left_length = left_end - left_start;
    int right_length = right_end - right_start;

    /* declare temporary lists */
    int left_half[left_length];
    int right_half[right_length];

    int r = 0; /* right_half index */
    int l = 0; /* left_half index */
    int i = 0; /* list index */

    /* copy left half of list into left_half */
    for(i = left_start; i < left_end; i++, l++)
        left_half[l] = list[i];

    /* copy right half of list into right_half */
    for(i = right_start; i < right_end; i++, r++)
        right_half[r] = list[i];
    
    /* merge left_half and right_half back into list */
    for(i = left_start, r = 0, l = 0; l < left_length && r < right_length; i++){
        if( left_half[l] < right_half[r] ) { list[i] = left_half[l]; }
        else { list[i] = right_half[r++]; }
    }

    /* Copy over leftovers of whichever temporary list hasn't finished */
    for( ; l < left_length; i++, l++) { list[i] = left_half[l]; }
    for( ; r < right_length; i++, r++) { list[i] = right_half[r]; }
}

void mergesort_iter(int left, int right, int *list)
{
    /* Base case, the list can be no simpler */
    if(right - left <= 1)
        return;

    /* set up bounds to slice array into */
    int left_start = left;
    int left_end = (left + right) / 2;
    int right_start = left_end;
    int right_end = right;

    /* sort left half */
    mergesort_iter(left_start, left_end, list);
    /* sort right half */
    mergesort_iter(right_start, right_end, list);

    /* merge sorted havles back together */
    merge(list, left_start, left_end, right_start, right_end);
} /* end merge sort recursive iterator */

void mergesort(int *list, int length)
{
    mergesort_iter(0, length, list);
} /* end merge sort */

int *allocate_array(int n, int random)
{
    int i, *v; /* v is pointer to the vector */

    v = (int*) malloc(n * sizeof(int));

    if(v == NULL) {
        printf("**Error in array allocation**\n");
        return (NULL);
    }

    /* initialize array with random values */
    switch( random ) {
        case 0:
            for(i = 0; i < n; i++)
                v[i] = 0;
            break;
        case 1:
            for(i = 0; i < n; i++)
                v[i] = (int) rand() % (MAXRAND - MINRAND + 1) + MINRAND;
            break;
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

