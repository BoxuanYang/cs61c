#include "transpose.h"
#include <stdlib.h>

#define MIN(a, b) ((a) < (b) ? (a) : (b))

/* The naive transpose function as a reference. */
void transpose_naive(int n, int blocksize, int *dst, int *src) {
    for (int x = 0; x < n; x++) {
        for (int y = 0; y < n; y++) {
            // dst[y][x] = src[x][y]
            dst[y + x * n] = src[x + y * n];
        }
    }
}

/* Implement cache blocking below. You should NOT assume that n is a
 * multiple of the block size. */
void transpose_blocking(int n, int blocksize, int *dst, int *src) {
    // YOUR CODE HERE

    for(int i = 0; i < n; i += blocksize){ // strating row index of submatrix
        for(int j = 0; j < n; j += blocksize){//starting col index of submatrix
            for(int ii = i; ii < MIN(i + blocksize, n); ii++){// row of submatrix
                for(int jj = j; jj < MIN(j + blocksize, n); jj++){                               
                    dst[jj + ii * n] = src[ii + jj * n];                   
                }
            }
        }
    }
    
}


