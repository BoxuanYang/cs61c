#include <stdio.h>
#include <x86intrin.h>



long long int sum(int *vals, int outer, int NUM_ELEMS) {


	long long int sum = 0;
	for(int w = 0; w < outer; w++) {
		for(unsigned int i = 0; i < NUM_ELEMS; i++) {
			if(vals[i] >= 128) {
				sum += vals[i];
			}
		}
	}
	
	return sum;
}


// *** TODO
long long int sum_simd(int *vals, int outer, int NUM_ELEMS) {

	__m128i _127 = _mm_set1_epi32(127);		// This is a vector with 127s in it... Why might you need this?
	long long int result = 0;				   // This is where you should put your final result!
	/* DO NOT DO NOT DO NOT DO NOT WRITE ANYTHING ABOVE THIS LINE. */

	for(int w = 0; w < outer; w++) {
		/* YOUR CODE GOES HERE */

		
		// vector_summ: the sum of one iteration of the array
		__m128i vector_summ = _mm_setzero_si128();

		// an array that stores the result of one whole itetation of array
	    unsigned int results[4] = {0};
		for(unsigned int i = 0; i+4 < NUM_ELEMS; i+=4){
			// load
			__m128i tmp = _mm_loadu_si128((__m128i*)&vals[i]);

			
			//Compare logic
			//Keep vecyor_a if >= 28, vector_tmp = 1..1 if > 127, 0 otherwise
			
		    __m128i mask = _mm_cmpgt_epi32(tmp, _127);

			tmp = _mm_and_si128(tmp, mask);

			//Addition logic
			
		    vector_summ = _mm_add_epi32(tmp, vector_summ);

		}

		_mm_storeu_si128((__m128i*)results, vector_summ);

		// You'll need a tail case. 
		for (int i = 4 * (NUM_ELEMS / 4); i < NUM_ELEMS; i++) {
			if (vals[i] >= 128) results[0] += vals[i];
		}

		for(int j = 0; j < 4; j++){
			result += results[j];
		}

	}

	return result;
}

int main(){
    int length = 5;
    int x[length];
    int outer = 2;

    for(int i = 0; i < length; i++){
        x[i] = 128 + i;
    }

    long long int result_sum = sum(x, outer, length);
    long long int result_simd_sum = sum_simd(x, outer, length);


}