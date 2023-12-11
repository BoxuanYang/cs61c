#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>





int main(){
    int size = 1644;
        
    //int thread_num = omp_get_num_threads();
    int chunks = size / 32;

    //printf("THread num is: %d \n \n", thread_num);

    printf("chunks is: %d \n", chunks);

    int index = chunks * 32;
    printf("index is: %d \n", index);

  

    return 0;
}