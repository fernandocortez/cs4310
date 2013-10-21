#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>

void *reduce(void *thread_count);
double *allocate_matrix(int n, int random);
double *free_matrix(double *v);
void print_matrix(double *v, int n);
int *calc_breakpoints(int n, int threads);
int *allocate_array(int n);
int *free_array(int *v);

#define MAXTHRDS 8

/* GLOBAL MEMORY */
double *matrix;
int *breakPoints;
int n; /* dimensions of matrix */
int thread_count; /* # of threads */

int main(int argc, char *argv[])
{
    long rank; /* looping variable */
    int i, array_size;
    double temp = 0.0;
    pthread_t *thread_handles;

    /* Check input arguments */
    switch( argc ) {
        case 3:
            n = atoi(argv[1]);
            if(n < 1) {
                printf("Matrix size must be greater than 1\n");
                exit(1); /* exit program with error */
            }

            thread_count = atoi(argv[2]);
            if(thread_count < 1) {
                printf("Thread count must be greater than 1\n");
                exit(1); /* exit program with error */
            }
            break;

        default:
            printf("**ERROR: Insufficient arguments**\n");
            printf("Usage: ./matmul [matrix size] [# threads]\n");
            exit(1); /* exit program with error */
    }

    thread_count = thread_count<MAXTHRDS ? thread_count : MAXTHRDS; /* caps # of threads spawned */
    n = n>thread_count ? n : thread_count; /* ensures at least 1 row per thread */
    array_size = n * n;

    /* Matrix memory allocation */
    matrix = allocate_matrix(array_size, 1);
    breakPoints = calc_breakpoints(n, thread_count);
    thread_handles = malloc(thread_count * sizeof(pthread_t));

    for(rank = 0; rank < thread_count; rank++)
        pthread_create(&thread_handles[rank], NULL, reduce, (void*) rank);

    for(rank = 0; rank < thread_count; rank++)
        pthread_join(thread_handles[rank], NULL);

    free(thread_handles);
    thread_handles = NULL;

    for(i = 0; i < thread_count; i++)
        temp += matrix[breakPoints[i] * n];
    matrix[0] = temp;

    if(n < 7)
        print_matrix(matrix, n);

    /* Deallocate memory */
    matrix = free_matrix(matrix);
    breakPoints = free_array(breakPoints);

    exit(0); /* exit program successfully */
}

void *reduce(void *rank)
{
    int i;
    double temp = 0.0;
    long my_rank = (long) rank;

    int rowband_start = breakPoints[my_rank] * n;
    int rowband_end = breakPoints[my_rank + 1] * n;

    for(i = rowband_start; i < rowband_end; i++) {
        temp += matrix[i];
    }

    matrix[rowband_start] = temp;

    return NULL;
}

double *allocate_matrix(int n, int random)
{
    int i;
    double *v; /* v is pointer to the vector */

    v = (double*) malloc(n * sizeof(double));

    if(v == NULL) {
        printf("**Error in matrix allocation: insufficient memory**\n");
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

    return (v); /* returns a pointer to null */
} /* end free matrix */

void print_matrix(double *v, int n)
{
    int row, column, start, end;

    for(row = 0; row < n; row++) {
        start = row * n;
        end = (row + 1) * n;
        for(column = start; column < end; column++)
            printf("%10.2f", v[column]);
        printf("\n");
    }
    printf("\n");
} /* end print matrix */

int *calc_breakpoints(int n, int threads)
{
    int i, *v; /* v is pointer to the vector */
    int rows = n/threads;
    int remainder = n%threads; /* number of rows not evenly distributed */
    int last_k = threads - remainder; /* number of threads get extra row */

    v = allocate_array(threads+1);

    /* row bands are attempted to be distributed evenly */
    for(i = 1; i <= threads; i++)
        v[i] += (v[i-1] + rows);

    /* The number of remaining rows are spread as evenly as possible amongst
     * the generated threads. This is done by giving the last k threads and
     * extra row, where k = remainder.
     */
    for(i = threads; i > last_k; i--)
        v[i] += remainder--;

    return(v); /* returns pointer to the vector */
} /* end calculate breakpoints */

int *allocate_array(int n)
{
    int i, *v; /* v is pointer to the vector */

    v = (int*) malloc(n * sizeof(int));

    if(v == NULL) {
        printf("**Error in array allocation: insufficient memory**\n");
        return (NULL);
    }

    for(i = 0; i < n; i++)
        v[i] = 0;

    return (v); /* returns pointer to the vector */
} /* end array allocation */

int *free_array(int *v)
{
    if(v == NULL)
        return (NULL);

    free(v);
    v = NULL;
    return(v); /* returns a pointer to null */
} /* end free array */

