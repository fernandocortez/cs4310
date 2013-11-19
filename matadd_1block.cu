/*
 * The purpose of this program is to add two matrices with a single thread
 * block on a GPU written in CUDA C.
 */

#include <stdio.h>
#include "util.h"

__global__ void matrixAdd(int *A, int *B, int *C, int size)
{
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int i;
    int temp;

    for(i = 0; i < size; i++) {
        temp = A[ty*size + i] + B[i*size + tx];
    }

    C[ty*size + tx] = temp;
}

int main(int argc, char *argv[])
{
    int i, j;
    int *matrixA;
    int *matrixB;
    int *matrixC;
    int *data;
    int size, total_size;

    if(argc != 2) {
        printf("format:%s [size of matrix]\n", argv[0]);
        exit(1);
    }

    size = atoi(argv[1]);
    total_size = size * size;

    /* allocate host memory */
    data = (int*) malloc(total_size * sizeof(int));

    /* allocate device memory */
    (cudaMalloc ((void**) &matrixA, sizeof(int) * total_size));
    (cudaMalloc ((void**) &matrixB, sizeof(int) * total_size));
    (cudaMalloc ((void**) &matrixC, sizeof(int) * total_size));
    checkError("Memory allocation");

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
    (cudaMemcpy( matrixA, data, sizeof(int)*total_size, cudaMemcpyHostToDevice ));
    (cudaMemcpy( matrixB, data, sizeof(int)*total_size, cudaMemcpyHostToDevice ));
    checkErrors("Memory copy 1");

    dim3 dimBlock(size, size);
    dim3 dimGrid(1, 1);

    /* timing */
    cudaEvent_t start_event, stop_event;
    cudaEventCreate(&start_event);
    cudaEventCreate(&stop_event);
    cudaEventRecord(start_event, 0);

    /* call kernel (global function) */
    matrixAdd<<dimGrid, dimBlock>>(matrixA, matrixB, matrixC, size);
    cudaThreadSynchronize();

    cudaEventRecord(stop_event, 0);
    cudaEventSynchronize(stop_event);
    float time_kernel;
    cudaEventElapsedTime(&time_kernel, start_event, stop_event);
    printf("Total time %f\n", time_kernel);

    /* copy data from device memory to host memory */
    (cudaMemcpy( data, matrixC, sizeof(int)*total_size, cudaMemcpyDeviceToHost ));
    checkErrors("Memory copy 2");

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
