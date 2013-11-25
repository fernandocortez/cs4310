/*
 * The purpose of this program is to add two matrices with multiple thread
 * blocks on a GPU written in CUDA C.
 */

#include <stdio.h>
#include "util.h"

__global__ void matrixAdd(float *A, float *B, float *C, size_t size, size_t width)
{
    size_t bx = blockIdx.x;
    size_t by = blockIdx.y;
    size_t tx = threadIdx.x;
    size_t ty = threadIdx.y;

    size_t row = by * width + ty;
    size_t column = bx * width + tx;
    size_t coordinate = row + column;
    int temp;

    temp = A[coordinate] + B[coordinate];
    C[coordinate] = temp;
}

int main(int argc, char *argv[])
{
    size_t i, j;
    size_t width;
    size_t size, total_size;
    int memory_size;

    float *matrixA = NULL;
    float *matrixB = NULL;
    float *matrixC = NULL;
    float *data = NULL;

    if(argc != 3) {
        printf("format:%s [size of matrix] [size of small matrix]\n", argv[0]);
        exit(1);
    }

    size = (unsigned) atoi(argv[1]);
    width = (unsigned) atoi(argv[2]);
    total_size = size * size;
    memory_size = total_size * sizeof(float);

    /* allocate host memory */
    data = (float*) malloc(memory_size);

    /* allocate device memory */
    (cudaMalloc( (void**) &matrixA, memory_size));
    (cudaMalloc( (void**) &matrixB, memory_size));
    (cudaMalloc( (void**) &matrixC, memory_size));
    checkErrors("Memory allocation\n");

    for(i = 0; i < total_size; i++)
        data[i] = 1.0; /* (int) (10 * rand()/32768.f); */

    if(size < 6) {
        for(i = 0; i < size; i++) {
            for(j = 0; j < size; j++)
                printf("%3.2f", data[i*size + j]);
            printf("\n");
        }
    }

    /* copy data from host memory to device memory */
    (cudaMemcpy( matrixA, data, memory_size, cudaMemcpyHostToDevice ));
    (cudaMemcpy( matrixB, data, memory_size, cudaMemcpyHostToDevice ));
    checkErrors("Memory copy 1\n");

    dim3 dimBlock(width, width);
    dim3 dimGrid(size/width, size/width);

    /* timing */
    cudaEvent_t start_event, stop_event;
    cudaEventCreate(&start_event);
    cudaEventCreate(&stop_event);
    cudaEventRecord(start_event, 0);

    /* call kernel (global function) */
    matrixAdd<<<dimGrid, dimBlock>>>(matrixA, matrixB, matrixC, size, width);
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
