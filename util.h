/** 
  *Utility functions. 
  */

#ifndef _MY_UTIL_H_
#define _MY_UTIL_H_

#include <stdlib.h>
#include <stdio.h>
#include <time.h> 

////////////////////////////////////////////////////////////////////////////////
// Constants
////////////////////////////////////////////////////////////////////////////////
/** Number of elements in the sequence */
 const int N = 9000000;

/** Minimum value of each element*/
const int EMIN = -10;

/** Maximum value of each element*/
const int EMAX = 10;

/** Minimum width constraints */
const int CMIN = 3;

/** Maximum width constraints */
const int CMAX = 5;

/** Maximum width of a sequence element */
const int MAXWIDTH = 5;

/** Partial block */
const int PARTIAL = 0;

/** No good partner */
const int NOPARTNER = -1;

/** Number of threads in each blcok*/
const int numThreadsPerBlock = 128;
//const int numThreadsPerBlock = 16;

////////////////////////////////////////////////////////////////////////////////
// DATA STRUCTURES
////////////////////////////////////////////////////////////////////////////////
/**
 * Pair structure. Each element of a sequence is a pair
 */
typedef struct Pair {
    /** Value of the pair */
	int value;
	/** Width of the pair*/
	int width;
} Pair;

/**
 * Block structure introduced in section 4.1. It is used to represent individual block information
 */
typedef struct{
	/**
	 * The start index of this block in the original sequence
	 */
	int startIndex;	
	/**
	 * The end index of this block in the original sequence
	 */
	int endIndex;	
	/**
	 * The elements of this block
	 */
	Pair *block;
	/**
	 * The right skew pointers of this block
	 */
	int *rsp;	
	/**
	 * The prefix right skew pointers of this block
	 */
	int *prsp;	
	/**
	 * The Tpp table for the whole block
	 */
	int **tpp;	
	/**
	 * The Tpj table for the whole block
	 */
	int **tpj;
} Block;

////////////////////////////////////////////////////////////////////////////////
// MEMORY ALLOCATION AND RECLAIM
///////////////////////////////////////////////////////////////////////////////
/** 
 * Allocate memories for a sequence. 
 * @param n The number of numbers to allocate
 * @return The allocated memory address
 */
struct Pair * allocatePairMem(int n)
{
	return (struct Pair *)malloc(n * sizeof(struct Pair));
}

/**
 * Allocate memories for a list of int numbers. 
 * @param n The number of int numbers to allocate
 * @return The allocated memory address
 */
int * allocateIntMem(int n)
{
	return (int *)malloc(n * sizeof(int));
}

/**
 * Allocate memories for a list of float numbers. 
 * @param n The number of float numbers to allocate
 * @return The allocated memory address
 */
float * allocateFloatMem(int n)
{
	return (float *)malloc(n * sizeof(float));
}

/**
 * Reclaim allocated memories
 * @param ptr The memory location
 */
void deallocate(void *ptr)
{
	free(ptr);
}

/**
 * error checking routine
 */
void checkErrors(char *label)
{
  // we need to synchronise first to catch errors due to
  // asynchroneous operations that would otherwise
  // potentially go unnoticed

  cudaError_t err;

  err = cudaThreadSynchronize();
  if (err != cudaSuccess)
  {
    char *e = (char*) cudaGetErrorString(err);
    fprintf(stderr, "CUDA Error: %s (at %s)", e, label);
  }

  err = cudaGetLastError();
  if (err != cudaSuccess)
  {
    char *e = (char*) cudaGetErrorString(err);
    fprintf(stderr, "CUDA Error: %s (at %s)", e, label);
  }
}

////////////////////////////////////////////////////////////////////////////////
// DATA SEQUENCE CREATION AND PRINT
///////////////////////////////////////////////////////////////////////////////

/**
 * Generate a list of integer values with given length.  Each integer value is generated randomly.
 * Each integer number is within the specified range;
 * !param n The number of int numbers in the list
 * !param min The minimum value among the generated values
 * !param max The maximum value among the generated values
 * !return The pointer of the list of generated int numbers. 
 *              NOTE: The pointer should be reclaimed after the use.
 */
int * generateRandomValues(int n, int min, int max)
{
    time_t seconds;
    time( &seconds );
    srand( (unsigned int)seconds );
	int *list = allocateIntMem( n );
	int i = 0;
	for (; i < n; i++)
	{
		list[i] = (rand() % (max - min + 1) + min);
	}
	return list;
}
/**
 * Generate a list of float values with given length.  Each float value is generated randomly.
 * Each float number is within the specified range;
 * !param n The number of float numbers in the list
 * !param min The minimum value among the generated values
 * !param max The maximum value among the generated values
 * !return The pointer of the list of generated int numbers. 
 *              NOTE: The pointer should be reclaimed after the use.
 */
float * generateFloatRandomValues(int n, int min, int max)
{
    time_t seconds;
    time( &seconds );
    srand( (unsigned int)seconds );
	float *list = allocateFloatMem( n );
	int i = 0;
	for (; i < n; i++)
	{
		list[i] = (rand() % (max - min + 1) + min);
	}
	return list;
}

/**
 * Construct sequence of number pairs. In each pair, the width is by default 1.
 * @param n The number of values
 * @param values The values of elements in the sequence
 * @return The sequence of pairs
 *         NOTE THAT, after the use, the sequence should be destroyed.
 */
struct Pair * constructPairs(int n, int values[]) {
    struct Pair *ptr = allocatePairMem( n );
	for (int i = 0; i < n; i++) {
        struct Pair temp = {values[i], 1};
        ptr[i] = temp;
	}
	return ptr;
}

/**
 * Construct sequence of number pairs. In each pair, the width is randomly generated.
 * @param n The number of values
 * @param values The values of elements in the sequence
 * @param maxWidth The maximum width of elements
 * @return The sequence of pairs
 *         NOTE THAT, after the use, the sequence should be destroyed.
 */
struct Pair * constructPairsWithRandomWidth(int n, int values[], int maxWidth) {
    time_t seconds;
    time( &seconds );
    srand( (unsigned int)seconds );
    struct Pair *ptr = allocatePairMem( n );
	for (int i = 0; i < n; i++) {
        struct Pair temp = {
               values[i], 
               (rand() % (maxWidth - 1 + 1) + 1)
        };
        ptr[i] = temp;
	}
	return ptr;
}

/**
 * Calculate the prefix sum (inclusive) of the given sequence
 * @param n The number of elements in the sequence
 * @param original The original sequence
 * @return The sequence representing the prefix sum of both values and widths
 */
struct Pair * prefixSumInclusive(int n, struct Pair * original)
{
    struct Pair *ptr = allocatePairMem( n );
    int j, value, width;
	for (int i = 0; i < n; i++) {
        j = 0; value = width = 0;
        while (j <= i)
        {
            value += original[j].value;
            width += original[j].width;
            j++;
        }
        struct Pair temp = {
               value,
               width
        };
        ptr[i] = temp;
	}
	return ptr;
}

/**
 * Print the specified data sequence.
 * @param n The number of data elements in the sequence
 * @param ptr Pointer to the data sequence
 */
__host__ void printPair(int n, Pair *ptr)
{
    printf("\n");
	printf("Value\t");
	int i;
	for (i = 0; i < n; i++) {
		printf( "%d\t", ptr[i].value );
	}
	printf("\n");
	printf("Width\t");
	for (i = 0; i < n; i++) {
		printf( "%d\t", ptr[i].width );
	}
	printf("\n");
	printf("Index\t");
	for (i = 0; i < n; i++) {
		printf( "%d\t", i);
	}
	printf("\n");
}

/**
 * Print the specified two-dimensional array
 * @param ptr Two-dimensional array
 * @param row Number of rows 
 * @param col Number of columns
 */
__host__ void printArray(int **ptr, int row, int col)
{
  for (int i = 0; i < row; i++)
  {
      printf("\n");
      for (int j = 0; j < col; j++)
      {
          printf("%d\t", ptr[i][j]);
      }
  }
}

/**
 * Print the specified two-dimensional array
 * @param ptr Two-dimensional array
 * @param row Number of rows 
 * @param col Number of columns
 */
__host__ void printArray(int *ptr, int col)
{
  printf("\n");
  for (int j = 0; j < col; j++)
  {
      printf("%d\t", ptr[j]);
  }
}

/**
 * Print the specified two-dimensional array
 * @param ptr Two-dimensional array
 * @param row Number of rows 
 * @param col Number of columns
 */
__host__ void printArray(float *ptr, int col)
{
  printf("\n");
  for (int j = 0; j < col; j++)
  {
      printf("%f\t", ptr[j]);
  }
}

////////////////////////////////////////////////////////////////////////////////
// Main routine -- Used for debug only
////////////////////////////////////////////////////////////////////////////////
int main1 (int argc, char **argv)
{
    int *test = generateRandomValues(N, EMIN, EMAX);
    //int i;
    //for (; i < n; i++)
    //{
    //    printf("%d\n", *test++);
    //} 
    struct Pair *ptr = constructPairsWithRandomWidth(N, test, MAXWIDTH);
    printPair(N, ptr);
    
    struct Pair *ps = prefixSumInclusive( N, ptr );
    printPair(N, ps);
    
    deallocate(ps);
    deallocate(ptr);
    deallocate(test);
    return 0;
}

#endif
