// suma_openmp.c
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define N 1048576

int main() {
    float *A = (float*) malloc(N * sizeof(float));
    float *B = (float*) malloc(N * sizeof(float));
    float *C = (float*) malloc(N * sizeof(float));

    // Inicialización
    for (int i = 0; i < N; i++) {
        A[i] = i * 0.5f;
        B[i] = i * 2.0f;
    }

    double start = omp_get_wtime();

    // Suma en paralelo
    #pragma omp parallel for
    for (int i = 0; i < N; i++) {
        C[i] = A[i] + B[i];
    }

    double end = omp_get_wtime();
    printf("Tiempo CPU (OpenMP): %f segundos\n", end - start);

    free(A); free(B); free(C);
    return 0;
}
