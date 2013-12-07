/*
 * The purpose of this program is to sort an array of integers using MPI
 * and random sampling.
 */

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <math.h>
#include <time.h>

void psrs(int *, int *, size_t, size_t, size_t *);
void swap_samples(int *, int *, size_t, size_t, size_t);
int intcmp(const void *, const void *);
int *allocate_array(size_t, int, char);
int *free_array(int *);
void print_array(int *, size_t, char);

#define MAXRAND 100
#define MINRAND 1

int main(int argc, char *argv[])
{
    /* VARIABLES */
    size_t m; /* size of initial array portion */
    size_t i; /* loop variable */
    int p; /* number of processors */
    int id; /* rank of processor */
    int *local_array;
    int *samples;

    /* PROGRAM START */
    if(argc != 2) {
        printf("**ERROR: Improper # of args\n");
        printf("Usage: mpirun -np [# procs] %s [size of array portion]\n", argv[0]);
        exit(1); /* exit program with error */
    }

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &p);
    MPI_Comm_rank(MPI_COMM_WORLD, &id);

    if(p < 2) {
        printf("Please input a larger number of processors\n");
        printf("Usage: mpirun -np [# procs] %s [size of array portion]\n", argv[0]);
        MPI_Finalize();
        exit(1); /* exit program with error */
    }

    m = (size_t) atoi(argv[1]);
    m = m < (size_t) p ? (size_t) p : m;
    srand((unsigned)time(NULL) + (unsigned) id); /* seeds random w/ respect to time */

    /* Memory allocation */
    local_array = allocate_array(m, 1, 'l');
    if(!id) { /* processor 0 will collect all samples */
        samples = allocate_array((size_t) p*(p-1), 0, 's');
    } else {
        samples = allocate_array((size_t) p-1, 0, 's');
    }

    /* Initial sorting */
    qsort(local_array, m, sizeof(int), intcmp);

    /* Test printing */
    if(m < 10)
        for(i = 0; i < p; i++) {
            MPI_Barrier(MPI_COMM_WORLD);
            if(id == i) {
                print_array(samples, p-1, 's');
                printf("This is process %d local array\n", id);
                print_array(local_array, m, (char) i);
                printf("\n");
            }
        }

    swap_samples(local_array, samples, (size_t) p, (size_t) id, m);
    //psrs(local_array, samples, (size_t) p, (size_t) id, &m);
    qsort(local_array, m, sizeof(int), intcmp);

    /* Test printing */
    for(i = 0; i < p; i++) {
        MPI_Barrier(MPI_COMM_WORLD);
        if(id == i && m < 10) {
            printf("This is process %d local array\n", id);
            print_array(local_array, m, (char) id);
            printf("\n");
        }
    }

    /* Deallocate memory */
    local_array = free_array(local_array);
    samples = free_array(samples);

    MPI_Finalize();
    exit(0); /* exit program successfully */
} /* end main */

void psrs(int *v, int *s, size_t p, size_t id, size_t *m)
{
    size_t i, n = *m * p;
    size_t sample_size = p-1;
    size_t ptr_count = p+1;
    int *sample_ptr[ptr_count];
    int inc_msg[sample_size][n];
    int out_msg[sample_size][n];
    size_t inc_msg_size[sample_size];
    size_t out_msg_size[sample_size];
    int temp;
    int source;
    int dest;
    MPI_Status status;

    for(i = 0; i < p; i++)
        sample_ptr[i] = v; /* all pointers point to first element of array */
    sample_ptr[p] = &v[*m-1]; /* last pointer at last element of array */
    for(i = 1; i < p; i++) {
        temp = s[i];
        while(*sample_ptr[i]++ < temp);
    }

    *m = sample_ptr[id+1] - sample_ptr[id];
    for(i = 0; i < p; i++) {
        if(id == i) {
        } else {
        }
    }

    for(i = 0; i < ptr_count; i++)
        sample_ptr[i] = NULL;
} /* end parallel sorting by sampling */

void swap_samples(int *v, int *s, size_t p, size_t id, size_t m)
{
    size_t i, step = m/p;
    size_t sample_size = p-1;
    int *msg;

    for(i = 1; i < p; i++)
        s[i-1] = v[i * step];

    /* all processors send samples to processor 0 */
    MPI_Gather(s, sample_size, MPI_INT, s, sample_size, MPI_INT, 0, MPI_COMM_WORLD);

    if(!id) {
        qsort(s, p * sample_size, sizeof(int), intcmp);
        /* isolates samples all processors will use */
        msg = allocate_array(sample_size, 0, 'm');
        for(i = 1; i < p; i++)
            msg[i-1] = s[i * sample_size];
        s = free_array(s);
        s = msg;
        msg = free_array(msg);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    /* broadcast samples to all processors */
    MPI_Bcast(s, sample_size, MPI_INT, 0, MPI_COMM_WORLD);
} /* end swap samples */

int intcmp(const void *p1, const void *p2)
{
    int x = *(int *)p1,
        y = *(int *)p2;

    return x <= y ? (x < y ? -1 : 0) : 1;
} /* end int compare */

int *allocate_array(size_t n, int random, char a)
{
    size_t i;
    int *v; /* v is pointer to the vector */

    v = (int*) malloc(n * sizeof(int));

    if(v == NULL) {
        printf("**Error in array %c allocation: insufficient memory**\n", a);
        return(NULL);
    }

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

    return (v); /* return pointer to vector to NULL */
} /* end free array */

void print_array(int *v, size_t n, char a)
{
    size_t i;

    for(i = 0; i < n; i++)
        printf("%6d", v[i]);
    printf("\n");
} /* end print array */

