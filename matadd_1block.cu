/*
 * The purpose of this program is to add two matrices with a single thread
 * block on a GPU written in CUDA C.
 */

#include <stdio.h>
#include "util.h"

__global__ void matrixAdd(int *A, int *B, int *C, size_t size)
{
    size_t tx = threadIdx.x;
    size_t ty = threadIdx.y;
    size_t coordinate = ty * size + tx;
    int temp;

    temp = A[coordinate] + B[coordinate];
    C[coordinate] = temp;
}

int main(int argc, char *argv[])
{
    size_t i, j;
    int *matrixA = NULL;
    int *matrixB = NULL;
    int *matrixC = NULL;
    int *data = NULL;
    size_t size, total_size;
    int memory_size;

    if(argc != 2) {
        printf("format:%s [size of matrix]\n", argv[0]);
        exit(1);
    }

    size = (unsigned) atoi(argv[1]);
    total_size = size * size;
    memory_size = total_size * sizeof(int);

    /* allocate host memory */
    data = (int*) malloc(memory_size);

    /* allocate device memory */
    (cudaMalloc( (void**) &matrixA, memory_size));
    (cudaMalloc( (void**) &matrixB, memory_size));
    (cudaMalloc( (void**) &matrixC, memory_size));
    checkErrors("Memory allocation\n");

    for(i = 0; i < total_size; i++)
        data[i] = 1; /* (int) (10 * rand()/32768.f); */

    if(size < 6) {
        for(i = 0; i < size; i++) {
            for(j = 0; j < size; j++)
                printf("%d", data[i*size + j]);
            printf("\n");
        }
    }

    /* copy data from host memory to device memory */
    (cudaMemcpy( matrixA, data, memory_size, cudaMemcpyHostToDevice ));
    (cudaMemcpy( matrixB, data, memory_size, cudaMemcpyHostToDevice ));
    checkErrors("Memory copy 1\n");

    dim3 dimBlock(size, size);
    dim3 dimGrid(1, 1);

    /* timing */
    cudaEvent_t start_event, stop_event;
    cudaEventCreate(&start_event);
    cudaEventCreate(&stop_event);
    cudaEventRecord(start_event, 0);

    /* call kernel (global function) */
    matrixAdd<<<dimGrid, dimBlock>>>(matrixA, matrixB, matrixC, size);
    cudaThreadSynchronize();

    cudaEventRecord(stop_event, 0);
    cudaEventSynchronize(stop_event);
    float time_kernel;
    cudaEventElapsedTime(&time_kernel, start_event, stop_event);
    printf("Total time %f\n", time_kernel);

    /* copy data from device memory to host memory */
    (cudaMemcpy( data, matrixC, memory_size, cudaMemcpyDeviceToHost ));
    checkErrors("Memory copy 2\n");

    if(size < 6) {
        for(i = 0; i < size; i++) {
            for(j = 0; j < size; j++)
                printf("%d", data[i*size + j]);
            printf("\n");
        }
    }

    free(data);
    data = NULL;
    cudaFree(matrixA);
    matrixA = NULL;
    cudaFree(matrixB);
    matrixB = NULL;
    cudaFree(matrixC);
    matrixC = NULL;

    exit(0);
}
