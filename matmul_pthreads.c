/*
 * This program multiplies two n by n matrices, A and B, producing a third
 * matrix, C. The matrices are multiplied using threads, each thread
 * producing a row-band of matrix C. The reduce, if not eliminate, read
 * contentions between the threads, matrix B will be partitioned. Once
 * each thread is finished using it partition, they will shuffle partitions.
 */

#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>

double *allocate_matrix(int n, int random, char m);
double *free_matrix(double *v);
void print_matrix(double *v, int n, char m);

int main(int argc, char *argv[])
{
    int n; /* dimensions of array */
    int array_size;
    double *matrixA;
    double *matrixB;
    double *matrixC;

    /* Check input arguments */
    switch( argc ) {
        case 3:
            n = atoi(argv[1]);
            if(n < 1) {
                printf("Matrix size must be greater than 1\n");
                exit(1); /* exit program with error */
            }
            array_size = n * n;
            break;

        default:
            printf("**ERROR: Insufficient arguments**\n");
            printf("Usage: ./matmul [matrix size] [# threads]\n");
            exit(1); /* exit program with error */
    }

    /* Matrix memory allocation */
    matrixA = allocate_matrix(array_size, 1, 'A');
    matrixB = allocate_matrix(array_size, 1, 'B');
    matrixC = allocate_matrix(array_size, 0, 'C');

    if(n < 7) {
        print_matrix(matrixA, n, 'A');
        print_matrix(matrixB, n, 'B');
        print_matrix(matrixC, n, 'C');
    }

    /* Deallocate memory */
    matrixA = free_matrix(matrixA);
    matrixB = free_matrix(matrixB);
    matrixC = free_matrix(matrixC);

    exit(0); /* exit program successfully */
}

double *allocate_matrix(int n, int random, char m)
{
    int i;
    double *v; /* v is pointer to the vector */

    v = (double*) malloc(n * sizeof(double));

    if(v == NULL) {
        printf("**Error in matrix %c allocation: insufficient memory**\n", m);
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

void print_matrix(double *v, int n, char m)
{
    int row, column, start, end;

    printf("matrix%c\n", m);
    for(row = 0; row < n; row++) {
        start = row * n;
        end = (row + 1) * n;
        for(column = start; column < end; column++)
            printf("%10.2f", v[column]);
        printf("\n");
    }
    printf("\n");
} /* end print matrix */

