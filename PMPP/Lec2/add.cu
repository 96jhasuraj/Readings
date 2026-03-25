#include<stdio.h>
#include <time.h>

__host__
void vec_add_cpu(float* a, float* b, float* c, int n) {
	for (int i = 0; i < n; i++) {
		c[i] = a[i] + b[i];
	}
}

__global__
void vec_add_kernel(float* a, float* b, float* c, int n) {
	int threadIndex = blockIdx.x * blockDim.x + threadIdx.x;
	if (threadIndex < n) {
		c[threadIndex] = a[threadIndex] + b[threadIndex];
	}
}

__host__
void vec_add_gpu(float* a, float* b, float* c, int n, int iters) {
	float* A, * B, * C;
	int size = sizeof(float) * n;

	cudaMalloc((void**)&A, size);
	cudaMalloc((void**)&B, size);
	cudaMalloc((void**)&C, size);

	cudaMemcpy(A, a, size, cudaMemcpyHostToDevice);
	cudaMemcpy(B, b, size, cudaMemcpyHostToDevice);
	float total = 0;
	for(int i=0;i<iters;i++) {
		clock_t start, end;
		start = clock();
		vec_add_kernel << <ceil(n / 128.0f), 128 >> > (A, B, C, n);
		cudaDeviceSynchronize();
		end = clock();
		//printf("CORE GPU Time:%f\n", (float)(end - start) / CLOCKS_PER_SEC);
		total += (float)(end - start) / CLOCKS_PER_SEC;
	}
	printf("CORE GPU Time:%f\n", total / iters);
	cudaMemcpy(c, C, size, cudaMemcpyDeviceToHost);
	cudaFree(A);
	cudaFree(B);
	cudaFree(C);

}
int main() {
	int n = 1000000;
	float* A, * B, * C;
	int size = sizeof(float) * n;
	A = (float*)malloc(size);
	B = (float*)malloc(size);
	C = (float*)malloc(size);
	
	for(int i=0;i<n;i++) {
		A[i] = i;
		B[i] = i;
		C[i] = 0.0f;
	}
	clock_t start, end;
	start = clock();

	vec_add_cpu(A, B, C, n);
	for (int i = 0; i < n; i++) {
		A[i] = i;
		B[i] = i;
		C[i] = 0.0f;
	}
	end = clock();
	printf("CPU Time:%f\n", (float)(end - start) / CLOCKS_PER_SEC);

	int ITERS = 10000;
	start = clock();
	vec_add_gpu(A, B, C, n, ITERS);
	end = clock();
	printf("GPU Time:%f\n", ((float)(end - start) /CLOCKS_PER_SEC))/ ITERS;

	printf("C[1]:%f",C[1]);
}


