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

/* Generates a random double between low and high */
double rand_double(double low, double high) {
    double range = (high - low);
    double div = RAND_MAX / range;
    return low + (rand() / div);
}

/* Generates a random matrix */
void rand_matrix(matrix *result, unsigned int seed, double low, double high) {
    srand(seed);
    for (int i = 0; i < result->rows; i++) {
        for (int j = 0; j < result->cols; j++) {
            set(result, i, j, rand_double(low, high));
        }
    }

    
}

/*
 * Returns the double value of the matrix at the given row and column.
 * You may assume `row` and `col` are valid. Note that the matrix is in row-major order.
 */
double get(matrix *mat, int row, int col) {
    // Task 1.1 TODO

    int cols = mat->cols;

    return mat->data[row * cols + col];
}

/*
 * Sets the value at the given row and column to val. You may assume `row` and
 * `col` are valid. Note that the matrix is in row-major order.
 */
void set(matrix *mat, int row, int col, double val) {
    // Task 1.1 TODO
    int cols = mat->cols;

    mat->data[row * cols + col] = val;

    return;
}

/*
 * Allocates space for a matrix struct pointed to by the double pointer mat with
 * `rows` rows and `cols` columns. You should also allocate memory for the data array
 * and initialize all entries to be zeros. `parent` should be set to NULL to indicate that
 * this matrix is not a slice. You should also set `ref_cnt` to 1.
 * You should return -1 if either `rows` or `cols` or both have invalid values. Return -2 if any
 * call to allocate memory in this function fails.
 * Return 0 upon success.
 */
int allocate_matrix(matrix **mat, int rows, int cols) {
    // Task 1.2 TODO
    // HINTS: Follow these steps.
    // 1. Check if the dimensions are valid. Return -1 if either dimension is not positive.
    if(rows <= 0 || cols <= 0) return -1;

    // 2. Allocate space for the new matrix struct. Return -2 if allocating memory failed.
    matrix *my_matrix = (struct matrix *) malloc(sizeof(struct matrix));
    if(my_matrix == NULL){
        free(my_matrix);
        return -2;
    }

    // 3. Allocate space for the matrix data, initializing all entries to be 0. Return -2 if allocating memory failed.
    my_matrix->data = (double *) malloc(cols * rows * sizeof(double));
    for(int i = 0; i < cols * rows; i++){
        my_matrix->data[i] = 0.0;
    }

    // 4. Set the number of rows and columns in the matrix struct according to the arguments provided.
    my_matrix->cols = cols;
    my_matrix->rows = rows;

    // 5. Set the `parent` field to NULL, since this matrix was not created from a slice.
    my_matrix->parent = NULL;

    // 6. Set the `ref_cnt` field to 1.
    my_matrix->ref_cnt = 1;

    // 7. Store the address of the allocated matrix struct at the location `mat` is pointing at.
    *mat = my_matrix;
    // 8. Return 0 upon success.
    return 0;
}

/*
 * You need to make sure that you only free `mat->data` if `mat` is not a slice and has no existing slices,
 * or that you free `mat->parent->data` if `mat` is the last existing slice of its parent matrix and its parent
 * matrix has no other references (including itself).
 */
void deallocate_matrix(matrix *mat) {
    // Task 1.3 TODO
    // HINTS: Follow these steps.
    // 1. If the matrix pointer `mat` is NULL, return.
    if(mat == NULL) return;

    // 2. If `mat` has no parent: decrement its `ref_cnt` field by 1. If the `ref_cnt` field becomes 0, then free `mat` and its `data` field.
    if(mat->parent == NULL){
        mat->ref_cnt--;

        if(mat->ref_cnt == 0){
            free(mat->data);
            free(mat);
        }
    }
    // 3. Otherwise, recursively call `deallocate_matrix` on `mat`'s parent, then free `mat`.
    else{
        deallocate_matrix(mat->parent);
        free(mat);
    }
}

/*
 * Allocates space for a matrix struct pointed to by `mat` with `rows` rows and `cols` columns.
 * Its data should point to the `offset`th entry of `from`'s data (you do not need to allocate memory)
 * for the data field. `parent` should be set to `from` to indicate this matrix is a slice of `from`
 * and the reference counter for `from` should be incremented. Lastly, do not forget to set the
 * matrix's row and column values as well.
 * You should return -1 if either `rows` or `cols` or both have invalid values. Return -2 if any
 * call to allocate memory in this function fails.
 * Return 0 upon success.
 * NOTE: Here we're allocating a matrix struct that refers to already allocated data, so
 * there is no need to allocate space for matrix data.
 */
int allocate_matrix_ref(matrix **mat, matrix *from, int offset, int rows, int cols) {
    // Task 1.4 TODO
    // HINTS: Follow these steps.
    // 1. Check if the dimensions are valid. Return -1 if either dimension is not positive.
    if(rows <= 0 || cols <= 0) return -1;

    // 2. Allocate space for the new matrix struct. Return -2 if allocating memory failed.
    matrix *new_matrix = (struct matrix *) malloc(sizeof(struct matrix));
    if(new_matrix == NULL) return -2;

    // 3. Set the `data` field of the new struct to be the `data` field of the `from` struct plus `offset`.
    new_matrix->data = from->data + offset;

    // 4. Set the number of rows and columns in the new struct according to the arguments provided.
    new_matrix->cols = cols;
    new_matrix->rows = rows;

    // 5. Set the `parent` field of the new struct to the `from` struct pointer.
    new_matrix->parent = from;

    // 6. Increment the `ref_cnt` field of the `from` struct by 1.
    from->ref_cnt++;

    // 7. Store the address of the allocated matrix struct at the location `mat` is pointing at.
    *mat = new_matrix;

    // 8. Return 0 upon success.
    return 0;
}

/*
 * Sets all entries in mat to val. Note that the matrix is in row-major order.
 */
void fill_matrix(matrix *mat, double val) {
    // Task 1.5 TODO


    int cols = mat->cols;
    int rows = mat->rows;

    int size = cols * rows;

  //  #pragma omp parallel
  //  {

   // int thread_id = omp_get_thread_num();
 //   int thread_num = omp_get_num_threads();

    // compute the size of each chunk 
  //  int chunk_size = size / thread_num;
    

    __m256d tmp = _mm256_set1_pd(val);  

   // int start = thread_id * chunk_size;
   // int end = (thread_id * chunk_size + chunk_size < size) ? thread_id * chunk_size + chunk_size : size;
    int i;
    for(i = 0; i+4 <= size; i+=4){
        _mm256_storeu_pd(mat->data + i, tmp);
    }

    while(i < size){
        mat->data[i] = val;
        i++;
    }

  //}

    return;
}

/*
 * Store the result of taking the absolute value element-wise to `result`.
 * Return 0 upon success.
 * Note that the matrix is in row-major order.
 */
int abs_matrix(matrix *result, matrix *mat) {
    // Task 1.5 TODO
    double *result_matrix = result->data;

    int cols = mat->cols;
    int rows = mat->rows;
    int size = cols * rows;

    __m256d minus_ones = _mm256_set1_pd (-1.0);
   // #pragma omp parallel
    //{

   // int thread_id = omp_get_thread_num();
   // int thread_num = omp_get_num_threads();

    // compute the size of each chunk 
  //  int chunk_size = size / thread_num;

    // compute the boundry of each thread
  //  int start = thread_id * chunk_size;
   // int end = (thread_id * chunk_size + chunk_size < size) ? thread_id * chunk_size + chunk_size : size;
    
    int i;
    for(i = 0; i+4 <= size; i+=4){
    //for(i = start; i+4 <= end; i+=4){
        // load a vector from mat
        __m256d vector = _mm256_loadu_pd(mat->data + i);

        // result_matrix[i] = (matrix[i] > 0) ? matrix[i] : -matrix[i];

        // multiply the vector with minus one
        __m256d vector_times_minus_one = _mm256_mul_pd(vector, minus_ones);

        // take the max vector
        __m256d tmp = _mm256_max_pd(vector, vector_times_minus_one);

        // store back
        _mm256_storeu_pd(result_matrix + i, tmp);
    }

    while(i < size){
        result_matrix[i] = (mat->data[i] > 0) ? mat->data[i] : -mat->data[i];
        i++;
    }

    //}

    return 0;
}

/*
 * (OPTIONAL)
 * Store the result of element-wise negating mat's entries to `result`.
 * Return 0 upon success.
 * Note that the matrix is in row-major order.
 */
int neg_matrix(matrix *result, matrix *mat) {
    // Task 1.5 TODO
    double *matrix = mat->data;
    double *result_matrix = result->data;
    int cols = mat->cols;
    int rows = mat->rows;

    for(int i = 0; i < rows * cols; i++){
        result_matrix[i] = -matrix[i];
    }

    return 0;
}

/*
 * Store the result of adding mat1 and mat2 to `result`.
 * Return 0 upon success.
 * You may assume `mat1` and `mat2` have the same dimensions.
 * Note that the matrix is in row-major order.
 */
int add_matrix(matrix *result, matrix *mat1, matrix *mat2) {
    // Task 1.5 TODO
    double *matrix1 = mat1->data;
    double *matrix2 = mat2->data;
    double *result_matrix = result->data;

    int cols = mat1->cols;
    int rows = mat1->rows;
    int size = rows * cols;

 //   #pragma omp parallel
   // {

  //  int thread_id = omp_get_thread_num();
  //  int thread_num = omp_get_num_threads();

    // compute the size of each chunk 
  //  int chunk_size = size / thread_num;

    // compute the boundry of each thread
  //  int start = thread_id * chunk_size;
  //  int end = (thread_id * chunk_size + chunk_size < size) ? thread_id * chunk_size + chunk_size : size;
    
    int i;
    for(i = 0; i+4 <= size; i+=4){
        // load 2 vectors from mat1 and mat2, respectively
        __m256d vector1 = _mm256_loadu_pd(matrix1 + i);
        __m256d vector2 = _mm256_loadu_pd(matrix2 + i);

        // add the 2 vectors
        __m256d summ = _mm256_add_pd(vector1, vector2);

        // store back
        _mm256_storeu_pd(result_matrix + i, summ);

        // result_matrix[i] = matrix1[i] + matrix2[i];        
    }

    while(i < size){
        result_matrix[i] = matrix1[i] + matrix2[i];  
        i++;
    }

 //   }

    return 0;
}

/*
 * (OPTIONAL)
 * Store the result of subtracting mat2 from mat1 to `result`.
 * Return 0 upon success.
 * You may assume `mat1` and `mat2` have the same dimensions.
 * Note that the matrix is in row-major order.
 */
int sub_matrix(matrix *result, matrix *mat1, matrix *mat2) {
    // Task 1.5 TODO
    double *matrix1 = mat1->data;
    double *matrix2 = mat2->data;
    double *result_matrix = result->data;

    int cols = mat1->cols;
    int rows = mat1->rows;

    for(int i = 0; i < rows * cols; i++){
        result_matrix[i] = matrix1[i] - matrix2[i];        
    }

    return 0;
}

/*
 * Store the result of multiplying mat1 and mat2 to `result`.
 * Return 0 upon success.
 * Remember that matrix multiplication is not the same as multiplying individual elements.
 * You may assume `mat1`'s number of columns is equal to `mat2`'s number of rows.
 * Note that the matrix is in row-major order.
 */
int mul_matrix(matrix *result, matrix *mat1, matrix *mat2) {
    // Assume mat1, mat2 and result are different
    // Task 1.6 TODO
    

    // Transpose mat2

    // This is a mat2->cols by mat2->rows matrix
    double *new_data = (double *) malloc(mat2->rows * mat2->cols * sizeof(double));

    for(int i = 0; i < mat2->cols; i++){
        for(int j = 0; j < mat2->rows; j++){
            // new_data[i][j] = mat2->data[j][i]
            new_data[i * mat2->rows + j] = mat2->data[j * mat2->cols + i];
        }
    }

    free(mat2->data);
    mat2->data = new_data;

    int tmp = mat2->rows;
    mat2->rows = mat2->cols;
    mat2->cols = tmp;


    int cols = result->cols;
    int rows = result->rows;

    int n = mat1->cols;

    /*#pragma omp parallel
    {

    int thread_id = omp_get_thread_num();
    int thread_num = omp_get_num_threads();

    // compute the size of each chunk 
    int chunk_size = rows / thread_num;

    // compute the boundry of each thread
    int start_rows = thread_id * chunk_size;
    int end_rows = (thread_id * chunk_size + chunk_size < rows) ? thread_id * chunk_size + chunk_size : rows;
    
    */
    int i;
    // Do the matrix multiplication
    for(i = 0; i < rows; i++){  // by row
        for(int j = 0; j < cols; j++){        // by column 

            int k;
            __m256d summ = _mm256_set1_pd(0.0);
            double results[4];

            //result_matrix[i][j] += matrix1[i][k] * matrix2[k][j];
            for(k = 0; k+4 <= n; k+=4){
                 // load 2 vectors from mat1 and mat2, respectively
                 __m256d vector1 = _mm256_loadu_pd(mat1->data + i * mat1->cols + k);
                 __m256d vector2 = _mm256_loadu_pd(mat2->data + j * mat2->cols + k);

                 // multiply these 2 vectors
                 __m256d product = _mm256_mul_pd(vector1, vector2);

                 // add the product to summ
                 summ = _mm256_add_pd(summ, product);
            }

            _mm256_storeu_pd(results, summ);

            while(k < n){
                //result_matrix[i][j] += matrix1[i][k] * matrix2[j][k];
                results[0] += mat1->data[i * mat1->cols + k] * mat2->data[j * mat2->cols + k];
                k++;
            }

            result->data[i * cols + j] = 0;
            for(int k = 0; k < 4; k++){
                result->data[i * cols + j] += results[k];
            }
        }
    }

    //}

    return 0;

}

/*
 * Store the result of raising mat to the (pow)th power to `result`.
 * Return 0 upon success.
 * Remember that pow is defined with matrix multiplication, not element-wise multiplication.
 * You may assume `mat` is a square matrix and `pow` is a non-negative integer.
 * Note that the matrix is in row-major order.
 */
int pow_matrix(matrix *result, matrix *mat, int pow) {
    // Task 1.6 TODO

    struct matrix *tmp = (struct matrix *) malloc(sizeof(struct matrix));
    tmp->rows = mat->rows;
    tmp->cols = mat->cols;
    tmp->data = (double *) malloc(mat->rows * mat->cols * sizeof(double));
    for(int i = 0; i < mat->rows * mat->cols; i++){
        tmp->data[i] = mat->data[i];
    }

    int sign = 0;
    for(int i = 0; i < pow - 1; i++){
        if(sign == 0){
            int mul_result = mul_matrix(result, tmp, mat);
            if(mul_result != 0) return -1;

            sign = 1;
        }

        else{
            int mul_result = mul_matrix(tmp, result, mat);
            if(mul_result != 0) return -1;
            sign = 0;
        }
        

        
    }

    if(sign == 0){
        for(int i = 0; i < mat->rows * mat->cols; i++){
            result->data[i] = tmp->data[i];
        }
    }

    free(tmp->data);
    free(tmp);

   

    return 0;
}
