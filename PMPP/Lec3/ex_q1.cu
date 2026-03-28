#include "timer.h"
#include <stdio.h>

/*
1. a generic matmul of shapes AxB BxC
2. modify kernel so that 1 kernel thread computes 1 row
*/

#define THREADS_PER_BLOCK 32
#define A_ROWS 2048
#define A_COLS 1024
#define B_ROWS A_COLS
#define B_COLS 2048

__global__
void mmGpuKernelPerThread(float* A, float* B, float* C, int ra, int cb, int cc) {
	int trow = blockDim.y * blockIdx.y + threadIdx.y;
	int tcol = blockDim.x * blockIdx.x + threadIdx.x;
	if (trow >= ra || tcol >= cc) return;
	C[trow * cc + tcol] = 0;
	for(int ind = 0;ind<cb;ind++) {
		C[trow * cc + tcol] += A[trow * cb + ind] * B[ind * cc + tcol];
	}
}

__global__
void mmGpuKernelPerRow(float* A, float* B, float* C, int ra, int cb, int cc) {
	int trow = blockDim.y * blockIdx.y + threadIdx.y;

	if (trow >= ra) return;
	for(int col = 0;col<cc;col++) {
		C[trow * cc + col] = 0;
	}
	for (int c = 0; c < cc; c++)
	{
		for (int ind = 0; ind < cb; ind++) {
			C[trow * cc + c] += A[trow * cb + ind] * B[ind * cc + c];
		}
	}
}

void mmGpuPerThread(float* a, float* b,float* c, int ra,int cb,int cc)
{
	// axb  X  bxc
	float* A, * B, * C;
	Timer timer;
	startTimer(&timer);
	cudaMalloc((void**)&A, ra* cb * sizeof(float));
	cudaMalloc((void**)&B, cb* cc * sizeof(float));
	cudaMalloc((void**)&C, ra* cc * sizeof(float));
	stopTimer(&timer);
	printElapsedTime(timer, "Memory allocation time: ");

	startTimer(&timer);

	cudaMemcpy(A, a, ra* cb * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(B, b, cb* cc * sizeof(float), cudaMemcpyHostToDevice);
	stopTimer(&timer);
	printElapsedTime(timer, "Memory copy time: ");

	dim3 tDim(THREADS_PER_BLOCK, THREADS_PER_BLOCK, 1);
	dim3 bDim((ra + tDim.x - 1) / tDim.x, (cc + tDim.y - 1) / tDim.y, 1);

	startTimer(&timer);
	mmGpuKernelPerThread << <bDim, tDim >> > (A, B, C, ra, cb,cc);
	cudaDeviceSynchronize();
	stopTimer(&timer);
	printElapsedTime(timer, "Core GPU mm time: mmGpuPerThread :");

	cudaMemcpy(c, C, ra* cc * sizeof(float), cudaMemcpyDeviceToHost);

	cudaFree(A);
	cudaFree(B);
	cudaFree(C);
}

void mmGpuPerRow(float* a, float* b, float* c, int ra, int cb, int cc)
{
	// axb  X  bxc
	float* A, * B, * C;
	Timer timer;
	startTimer(&timer);
	cudaMalloc((void**)&A, ra * cb * sizeof(float));
	cudaMalloc((void**)&B, cb * cc * sizeof(float));
	cudaMalloc((void**)&C, ra * cc * sizeof(float));
	stopTimer(&timer);
	printElapsedTime(timer, "Memory allocation time: ");

	startTimer(&timer);

	cudaMemcpy(A, a, ra * cb * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(B, b, cb * cc * sizeof(float), cudaMemcpyHostToDevice);
	stopTimer(&timer);
	printElapsedTime(timer, "Memory copy time: ");

	dim3 tDim(1, 1, 1);
	dim3 bDim(ra, 1, 1);

	startTimer(&timer);
	mmGpuKernelPerRow << <bDim, tDim >> > (A, B, C, ra, cb, cc);
	cudaDeviceSynchronize();
	stopTimer(&timer);
	printElapsedTime(timer, "Core GPU mm time: mmGpuPerRow :");

	cudaMemcpy(c, C, ra * cc * sizeof(float), cudaMemcpyDeviceToHost);

	cudaFree(A);
	cudaFree(B);
	cudaFree(C);
}
void mmCpu(float* a, float* b, float* c, int ra, int rb,int cc) {
	// axb  X  bxc
	for (int i = 0; i < ra; i++) {
		for (int j = 0; j < cc; j++) {
			c[i * cc + j] = 0.0f;
			for (int k = 0; k < rb; k++) {
				c[i * cc + j] += a[i * rb + k] * b[k * cc + j];
			}
		}
	}
}

void checkAB(float* a, float* b, int size)
{
	for (int i = 0; i < size; i++)
	{
		if (a[i] != b[i]) {
			printf("Error at %d: %f != %f\n", i, a[i], b[i]);
			return;
		}
	}
	printf("MM successfull \n");
}

int main()
{
	float* A, * B, * C , *C_pertPoint,*C_perRow,*C_perCol;
	A = (float*)malloc(sizeof(float) * A_COLS * A_ROWS);
	B = (float*)malloc(sizeof(float) * A_COLS * B_COLS);
	C = (float*)malloc(sizeof(float) * B_COLS * A_ROWS);
	C_pertPoint = (float*)malloc(B_COLS * A_ROWS * sizeof(float));
	C_perRow = (float*)malloc(B_COLS * A_ROWS * sizeof(float));
	C_perCol = (float*)malloc(B_COLS * A_ROWS * sizeof(float));
	for (int i = 0; i < A_ROWS * A_COLS; i++) {
		A[i] = rand() % 100;
	}
	for (int i = 0; i < B_ROWS * B_COLS; i++) {
		B[i] = rand() % 100;
	}
	for (int i = 0; i < A_ROWS * B_COLS; i++) {
		C[i] = 0.0;
	}
	Timer timer;

	startTimer(&timer);
	mmCpu(A, B, C, A_ROWS, A_COLS, B_COLS);
	stopTimer(&timer);
	printElapsedTime(timer, "CPU mm time: ");

	startTimer(&timer);
	mmGpuPerThread(A, B,C_pertPoint, A_ROWS, A_COLS,B_COLS);
	stopTimer(&timer);
	printElapsedTime(timer, "GPU mm total time: per Point");
	checkAB(C, C_pertPoint, A_ROWS * B_COLS);

	startTimer(&timer);
	mmGpuPerThread(A, B, C_perRow, A_ROWS, A_COLS, B_COLS);
	stopTimer(&timer);
	printElapsedTime(timer, "GPU mm total time: per Row");
	checkAB(C, C_perRow, A_ROWS * B_COLS);


	free(C_perCol);
	free(C_perRow);
	free(A);
	free(B);
	free(C);
	free(C_pertPoint);

}