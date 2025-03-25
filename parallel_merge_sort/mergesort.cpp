/*
 *  Parallel MergeSort implementation using OpenMP
 *  Students:
 *  - Capodanno Mario
 *  - Grillo Valerio
 *  - Lovino Emanuele
 */

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>


#include <algorithm>
#include <iostream>


#include <cstdio>
#include <cstdlib>


#include <cmath>
#include <cstring>
#include <ctime>


#include "omp.h"

#define THREASHOLD 10000000
#define REDUCING_FACTOR 1000
#define DEFAULT_SIZE 10000
#define SORT_CUTOFF 6
#define THREAD_NUM 8

/**
 * helper routine: check if array is sorted correctly
 */
bool isSorted(int ref[], int data[], const size_t size) {
  std::sort(ref, ref + size);
  for (size_t idx = 0; idx < size; ++idx) {
    if (ref[idx] != data[idx]) {
      return false;
    }
  }
  return true;
}

/*
 *sequential merge step (straight-forward implementation)
 */
void MsMergeSequential(int *out, int *in, long begin1, long end1, long begin2,
                       long end2, long outBegin) {
  long left = begin1;
  long right = begin2;

  long idx = outBegin;

  while (left < end1 && right < end2) {
    if (in[left] <= in[right]) {
      out[idx] = in[left];
      left++;
    } else {
      out[idx] = in[right];
      right++;
    }
    idx++;
  }

  while (left < end1) {
    out[idx] = in[left];
    left++, idx++;
  }

  while (right < end2) {
    out[idx] = in[right];
    right++, idx++;
  }
}
/*
   *Parallel Merge step of two arrays A, B with load balancing.
   *Parameters for the two recursive calls are computed through:
         - index of the median element of A (half1)
         - k-th index of the first element in B greater
   *Implemented a cut-off mechanism based on arrays size, switching to the
   sequential merge.
*/
void MsMergeParallel(int *out, int *in, long begin1, long end1, long begin2, long end2, long outBegin, long merge_cutoff = 100000) {
  long size = end1 - begin1 + end2 - begin2;
  
  // base case
  if (size >= THREASHOLD) {
    merge_cutoff = size / REDUCING_FACTOR;
  } else {
    merge_cutoff = DEFAULT_SIZE;
  }

  // base case
  if (size <= merge_cutoff) {
    MsMergeSequential(out, in, begin1, end1, begin2, end2, outBegin);
  } 
  else {

    // load balancing necessary if the second portion of the array is bigger
    // than the first one
    if (end1 - begin1 < end2 - begin2) {
      std::swap(begin1, begin2);
      std::swap(end1, end2);
    }
    // find the middle position index of the first array
    long half1 = begin1 + (end1 - begin1) / 2;

    // find the correspondent nearest element in the second part of the array
    // through a binary search
    long half2 =
        (long)(std::lower_bound(in + begin2, in + end2, in[half1]) - in);

	#pragma omp task shared(out, in)
		MsMergeParallel(out, in, begin1, half1, begin2, half2, outBegin);

	#pragma omp task shared(out, in)
		MsMergeParallel(out, in, half1, end1, half2, end2,
						outBegin + (half1 - begin1) + (half2 - begin2));

	#pragma omp taskwait
  }

}

/*
 *Recursive division of array[] using tmp[] as temporary buffer.
 *Boolean inplace logic allows an efficient use of the available space.
 *Uses depth parameter as cut-off mechanism.
 *Through the final clause, this function can be used for both parallel and sequential partition avoiding redundancy
 */

void MsPartition(int *array, int *tmp, bool inplace, long begin, long end,
                 long depth) {

  if (depth <= SORT_CUTOFF) {
    // Base case: if the segment is small enough, use std::sort directly
    if (inplace) {
      std::sort(array + begin, array + end);
    } else {
      std::copy(array + begin, array + end, tmp + begin);
      std::sort(tmp + begin, tmp + end);
    }
    return;
  }

  if (begin < (end - 1)) {
    const long half = (begin + end) / 2;

	// Recursive case
	#pragma omp task shared(array, tmp) firstprivate(begin, half, end, depth)      \
		final(depth <= 13)
		MsPartition(array, tmp, !inplace, begin, half, depth - 1);

	#pragma omp task shared(array, tmp) firstprivate(begin, half, end, depth)      \
		final(depth <= 13)
		MsPartition(array, tmp, !inplace, half, end, depth - 1);

	#pragma omp taskwait

	// Merge the two sorted halves together
    if (inplace) {
      MsMergeParallel(array, tmp, begin, half, half, end, begin);
    } else {
      MsMergeParallel(tmp, array, begin, half, half, end, begin);
    }
  } 
  else if (!inplace) {
    tmp[begin] = array[begin];
  }
}

/*
 *Takes array[] and tmp[] of the same size, compute maxDepth based on input size,  
 spawns threads and calls recursive partition step.
 */
void MergeSort(int *array, int *tmp, const size_t size) {
  int depth = log2(size);

  #pragma omp parallel num_threads(THREAD_NUM)
  {
  	#pragma omp single nowait
    MsPartition(array, tmp, true, 0, size, depth);
  }
}

/**
 * @brief program entry point
 */
int main(int argc, char *argv[]) {
  // variables to measure the elapsed time
  struct timeval t1, t2;
  double etime;

  // expect one command line arguments: array size
  if (argc != 2) {
    printf("Usage: MergeSort.exe <array size> \n");
    printf("\n");
    return EXIT_FAILURE;
  } else {
    const size_t stSize = strtol(argv[1], NULL, 10);
    int *data = (int *)malloc(stSize * sizeof(int));
    int *tmp = (int *)malloc(stSize * sizeof(int));
    int *ref = (int *)malloc(stSize * sizeof(int));

    printf("Initialization...\n");

    srand(95);
    for (size_t idx = 0; idx < stSize; ++idx) {
      data[idx] = (int)(stSize * (double(rand()) / RAND_MAX));
    }
    std::copy(data, data + stSize, ref);

    double dSize = (stSize * sizeof(int)) / 1024 / 1024;
    printf("Sorting %zu elements of type int (%f MiB)...\n", stSize, dSize);

    gettimeofday(&t1, NULL);
    MergeSort(data, tmp, stSize);
    gettimeofday(&t2, NULL);

    etime = (t2.tv_sec - t1.tv_sec) * 1000 + (t2.tv_usec - t1.tv_usec) / 1000;
    etime = etime / 1000;

    printf("done, took %f sec. Verification...", etime);
    if (isSorted(ref, data, stSize)) {
      printf(" successful.\n");
    } else {
      printf(" FAILED.\n");
    }

    free(data);
    free(tmp);
    free(ref);
  }

  return EXIT_SUCCESS;
}