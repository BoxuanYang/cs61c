#include "matrix.h"
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

// Include SSE intrinsics
#if defined(_MSC_VER)
#include <intrin.h>
#elif defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))
#include <immintrin.h>
#include <x86intrin.h>
#endif

/* Below are some intel intrinsics that might be useful
 * void _mm256_storeu_pd (double * mem_addr, __m256d a)
 * __m256d _mm256_set1_pd (double a)
 * __m256d _mm256_set_pd (double e3, double e2, double e1, double e0)
 * __m256d _mm256_loadu_pd (double const * mem_addr)
 * __m256d _mm256_add_pd (__m256d a, __m256d b)
 * __m256d _mm256_sub_pd (__m256d a, __m256d b)
 * __m256d _mm256_fmadd_pd (__m256d a, __m256d b, __m256d c)
 * __m256d _mm256_mul_pd (__m256d a, __m256d b)
 * __m256d _mm256_cmp_pd (__m256d a, __m256d b, const int imm8)
 * __m256d _mm256_and_pd (__m256d a, __m256d b)
 * __m256d _mm256_max_pd (__m256d a, __m256d b)
*/

/*
 * Generates a random double between `low` and `high`.
 */
double rand_double(double low, double high) {
    double range = (high - low);
    double div = RAND_MAX / range;
    return low + (rand() / div);
}

/*
 * Generates a random matrix with `seed`.
 */
void rand_matrix(matrix *result, unsigned int seed, double low, double high) {
    srand(seed);
    for (int i = 0; i < result->rows; i++) {
        for (int j = 0; j < result->cols; j++) {
            set(result, i, j, rand_double(low, high));
        }
    }
}

/*
 * Allocate space for a matrix struct pointed to by the double pointer mat with
 * `rows` rows and `cols` columns. You should also allocate memory for the data array
 * and initialize all entries to be zeros. Remember to set all fieds of the matrix struct.
 * `parent` should be set to NULL to indicate that this matrix is not a slice.
 * You should return -1 if either `rows` or `cols` or both have invalid values, or if any
 * call to allocate memory in this function fails. If you don't set python error messages here upon
 * failure, then remember to set it in numc.c.
 * Return 0 upon success and non-zero upon failure.
 */
int allocate_matrix(matrix **mat, int rows, int cols) {
    /* TODO: YOUR CODE HERE */

    mat = (struct matrix**) malloc(sizeof(struct matrix *));
    if(mat == NULL) return -1;

    // 分配内存给指针所指向的结构体
    *mat = (struct matrix*) malloc(sizeof(struct matrix));
    if(*mat == NULL) return -1;

    // 设定行列数
    (*mat)->rows = rows;
    (*mat)->cols = cols;

    // 为data字段分配内存。data是一个二维数组，因此需要两步
    (*mat)->data = (double **) malloc(rows * sizeof(double**));
    if((*mat)->data == NULL) {
        // 处理内存分配失败
        free(*mat);
        free(mat);
        return -1;
    }

    for(int i = 0; i < rows; i++) {
        (*mat)->data[i] = (double *) malloc(cols * sizeof(double));
        if((*mat)->data[i] == NULL) {
            // 处理内存分配失败
            for(int j = 0; j < i; j++) {
                free((*mat)->data[j]);
            }
            free((*mat)->data);
            free(*mat);
            free(mat);
            return -1;
        }
    }

    

    return 0;


}

/*
 * Allocate space for a matrix struct pointed to by `mat` with `rows` rows and `cols` columns.
 * This is equivalent to setting the new matrix to be
 * from[row_offset:row_offset + rows, col_offset:col_offset + cols]
 * If you don't set python error messages here upon failure, then remember to set it in numc.c.
 * Return 0 upon success and non-zero upon failure.
 */
int allocate_matrix_ref(matrix **mat, matrix *from, int row_offset, int col_offset,
                        int rows, int cols) {
    /* TODO: YOUR CODE HERE */

    int allocation_result = allocate_matrix(mat, rows, cols);

    if(allocation_result != 0) return -1;

    double **from_matrix = from->data;
    double **mat_matrix = (*mat)->data;

    for(int i = 0; i < rows; i++){
        for(int j = 0; j < cols; j++){
            mat_matrix[i][j] = from_matrix[i + row_offset][j + col_offset];
        }
    }

    return 0;
}

/*
 * This function will be called automatically by Python when a numc matrix loses all of its
 * reference pointers.
 * You need to make sure that you only free `mat->data` if no other existing matrices are also
 * referring this data array.
 * See the spec for more information.
 */
void deallocate_matrix(matrix *mat) {
    /* TODO: YOUR CODE HERE */
    int rows = mat->rows;

    if(mat->ref_cnt != 1) return;

    for(int i = 0; i < rows; i++){
        free(mat->data[i]);
    }

    free(mat);

}

/*
 * Return the double value of the matrix at the given row and column.
 * You may assume `row` and `col` are valid.
 */
double get(matrix *mat, int row, int col) {
    /* TODO: YOUR CODE HERE */
    double **data = mat->data;

    return data[row][col];
}

/*
 * Set the value at the given row and column to val. You may assume `row` and
 * `col` are valid
 */
void set(matrix *mat, int row, int col, double val) {
    /* TODO: YOUR CODE HERE */
    double **data = mat->data;

    data[row][col] = val;
}

/*
 * Set all entries in mat to val
 */
void fill_matrix(matrix *mat, double val) {
    /* TODO: YOUR CODE HERE */

    int rows = mat->rows;
    int cols = mat->cols;
    double **data = mat->data;

    for(int i = 0; i < rows; i++){
        for(int j = 0; j < cols; j++)
            data[i][j] = val;       
    }
}

/*
 * Store the result of adding mat1 and mat2 to `result`.
 * Return 0 upon success and a nonzero value upon failure.
 */
int add_matrix(matrix *result, matrix *mat1, matrix *mat2) {
    /* TODO: YOUR CODE HERE */
    double **matrix1 = mat1->data;
    double **matrix2 = mat2->data;
    double **result_matrix = result->data;

    if(mat1->rows != mat2->rows || mat1->cols != mat2->cols 
    || result->rows != mat1->rows || result->cols != mat1->cols){
        return -1;
    }

    int rows = mat1->rows;
    int cols = mat2->cols;

    for(int i = 0; i < rows; i++){
        for(int j = 0; j < cols; j++)
            result_matrix[i][j] = matrix1[i][j] + matrix2[i][j];       
    }

    return 0;
}

/*
 * Store the result of subtracting mat2 from mat1 to `result`.
 * Return 0 upon success and a nonzero value upon failure.
 */
int sub_matrix(matrix *result, matrix *mat1, matrix *mat2) {
    /* TODO: YOUR CODE HERE */
    double **matrix1 = mat1->data;
    double **matrix2 = mat2->data;
    double **result_matrix = result->data;

    if(mat1->rows != mat2->rows || mat1->cols != mat2->cols 
    || result->rows != mat1->rows || result->cols != mat1->cols){
        return -1;
    }

    int rows = mat1->rows;
    int cols = mat2->cols;

    for(int i = 0; i < rows; i++){
        for(int j = 0; j < cols; j++)
            result_matrix[i][j] = matrix1[i][j] - matrix2[i][j];       
    }

    return 0;
}

/*
 * Store the result of multiplying mat1 and mat2 to `result`.
 * Return 0 upon success and a nonzero value upon failure.
 * Remember that matrix multiplication is not the same as multiplying individual elements.
 */
int mul_matrix(matrix *result, matrix *mat1, matrix *mat2) {
    /* TODO: YOUR CODE HERE */
    double **matrix1 = mat1->data;
    double **matrix2 = mat2->data;
    double **result_matrix = result->data;

    int m = result->rows;
    int n = result->cols;

    if(m != mat1->rows|| n != mat2->cols || mat1->cols != mat2->rows)
        return -1;
    

    
    for(int i = 0; i < m; i++){
        for(int k = 0; k < mat1->cols; k++){
        //for(int j = 0; j < n; j++){
            for(int j = 0; j < n; j++){
            //for(int k = 0; k < mat1->cols; k++){
                result_matrix[i][j] += matrix1[i][k] * matrix2[k][j];
            }
        }
    }

    return 0;
    
}

/*
 * Store the result of raising mat to the (pow)th power to `result`.
 * Return 0 upon success and a nonzero value upon failure.
 * Remember that pow is defined with matrix multiplication, not element-wise multiplication.
 */
int pow_matrix(matrix *result, matrix *mat, int pow) {
    /* TODO: YOUR CODE HERE */
    if(result->rows != mat->rows || result->cols != mat->cols || pow <= 0)
        return -1;
    
    for(int i = 0; i < pow; i++){
        int mul_result;
        mul_result = mul_matrix(result, mat, mat);

        if(mul_result != 0) return -1;
    }

    return 0;
}

/*
 * Store the result of element-wise negating mat's entries to `result`.
 * Return 0 upon success and a nonzero value upon failure.
 */
int neg_matrix(matrix *result, matrix *mat) {
    /* TODO: YOUR CODE HERE */
    double **result_matrix = result->data;
    double **matrix = mat->data;


    if(result->rows != mat->rows || result->cols != mat->cols)
        return -1;
    
    int rows = mat->rows;
    int cols = mat->cols;

    for(int i = 0; i < rows; i++){
        for(int j = 0; j < cols; j++){
            result_matrix[i][j] = 0 - matrix[i][j];
        }       
    }

    return 0;
}

/*
 * Store the result of taking the absolute value element-wise to `result`.
 * Return 0 upon success and a nonzero value upon failure.
 */
int abs_matrix(matrix *result, matrix *mat) {
    /* TODO: YOUR CODE HERE */
    if(result->rows != mat->rows || result->cols != mat->cols)
        return -1;
    
    int rows = mat->rows;
    int cols = mat->cols;

    double **matrix = mat->data;
    double **result_matrix = result->data;

    for(int i = 0; i < rows; i++){
        for(int j = 0; j < cols; j++){
            double val = (matrix[i][j] < 0) ? -matrix[i][j] : matrix[i][j];
            result_matrix[i][j] = val;
        }       
    }

    return 0;

}
