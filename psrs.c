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
        exit(1);
    }

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &p);
    MPI_Comm_rank(MPI_COMM_WORLD, &id);

    if(p < 2) {
        printf("Please input a larger number of processors\n");
        MPI_Finalize();
        exit(1);
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
    swap_samples(local_array, samples, (size_t) p, (size_t) id, m);

    /* Test printing */
    if(m < 7)
        for(i = 0; i < p; i++) {
            MPI_Barrier(MPI_COMM_WORLD);
            if(id == i) {
                print_array(samples, p-1, 's');
                printf("This is process %d local array\n", id);
                print_array(local_array, m, (char) i);
                printf("\n");
            }
        }

    psrs(local_array, samples, (size_t) p, (size_t) id, &m);

    /* Test printing */
    if(m < 7)
        for(i = 0; i < p; i++) {
            MPI_Barrier(MPI_COMM_WORLD);
            if(id == i) {
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
    size_t i, sample_size = p-1;
    int **sample_ptr = (int**) malloc(sample_size * sizeof(int*));
    int temp;

    for(i = 0; i < sample_size; i++)
        sample_ptr[i] = v;
    for(i = 1; i < sample_size; i++) {
        temp = s[i];
        while(*sample_ptr[i]++ < temp);
    }
    for(i = 0; i < p; i++) {
        if(id == i) {
            i
        } else {
        }
    }

    /* Deallocate memory */
    for(i = 0; i < sample_size; i++)
        sample_ptr[i] = NULL;
    free(sample_ptr);
    sample_ptr = NULL;
} /* end parallel sorting by sampling */

void swap_samples(int *v, int *s, size_t p, size_t id, size_t m)
{
    size_t i, step = m/p, temp;
    int *msg;
    MPI_Status status;

    if(!id)
        for(i = 0; i < p-1; i++)
            s[i] = v[i * step];

    MPI_Barrier(MPI_COMM_WORLD);

    if(!id) {
        temp = p * (p-1);
        /* p0 receives samples from other processors */
        for(; i < temp; i++)
            MPI_Recv(&s[i], 1, MPI_INT, i, i, MPI_COMM_WORLD, &status);

        qsort(s, temp, sizeof(int), intcmp);
        
        /* isolates samples all processors will use */
        msg = allocate_array(p-1, 0, 'm');
        step = p-1;
        temp = step;
        for(i = 1; i < p; i++)
            msg[i] = s[i * step];
        for(i = 1; i < p; i++)
            MPI_Send(msg, temp, MPI_INT, i, i, MPI_COMM_WORLD);
        s = free_array(s);
        s = msg;
        msg = free_array(msg);
    } else {
        /* send samples to p0 */
        for(i = 1; i < p; i++)
            MPI_Send(&v[i*step], 1, MPI_INT, 0, id, MPI_COMM_WORLD);
        temp = p-1;
        MPI_Recv(s, temp, MPI_INT, 0, id, MPI_COMM_WORLD, &status);
    }
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

