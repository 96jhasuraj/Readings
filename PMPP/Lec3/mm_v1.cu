#include "timer.h"
#include <stdio.h>

__global__
void mmGpuKernel(float* a, float* b, float* c, int rm, int cm){
	int row = blockDim.y * blockIdx.y + threadIdx.y;
	int col = blockDim.x * blockIdx.x + threadIdx.x;
	if (row >= rm || col >= cm) return;
	
	unsigned int idx = row * cm + col;
	c[idx] = 0;
	for (int i = 0; i < cm; i++)
	{
		c[idx] += a[row * cm + i] * b[i * cm + col];
	}
}


void mmGpu(float* a, float* b, float* c, int rm, int cm) {
	float* A, * B, * C;
	Timer timer;
	startTimer(&timer);
	cudaMalloc((void**)&A, rm * cm * sizeof(float));
	cudaMalloc((void**)&B, rm * cm * sizeof(float));
	cudaMalloc((void**)&C, rm * cm * sizeof(float));
	stopTimer(&timer);
	printElapsedTime(timer, "Memory allocation time: ");

	startTimer(&timer);

	cudaMemcpy(A, a, rm * cm * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(B, b, rm * cm * sizeof(float), cudaMemcpyHostToDevice);
	stopTimer(&timer);
	printElapsedTime(timer, "Memory copy time: ");

	dim3 tDim(32, 32, 1);
	dim3 bDim((rm + tDim.x - 1) / tDim.x, (cm + tDim.y - 1) / tDim.y, 1);

	startTimer(&timer);
	mmGpuKernel << <bDim,tDim>>>(A, B, C, rm, cm);
	cudaDeviceSynchronize();
	stopTimer(&timer);
	printElapsedTime(timer, "Core GPU mm time: ");

	cudaMemcpy(c, C, rm * cm * sizeof(float), cudaMemcpyDeviceToHost);
	
	cudaFree(A);
	cudaFree(B);
	cudaFree(C);
}

void checkAB(float* a, float* b , int size)
{
	for (int i = 0; i < size; i++)
	{
		if(a[i]!= b[i]){
			printf("Error at %d: %f != %f\n", i, a[i], b[i]);
			return;
		}
	}
	printf("MM successfull \n");
}
void mmCpu(float* a, float* b, float* c, int rm, int cm) {
	for(int i = 0; i < rm; i++){
		for(int j = 0; j < cm; j++){
			c[i * cm + j] = 0.0f;
			for(int k = 0; k < cm; k++){
				c[i * cm + j] += a[i * cm + k] * b[k * rm + j];
			}
		}
	}
}

int main()
{
	float* A, * B, * C , *C_D;
	int rm = 1000, cm = 1000;
	
	A = (float*)malloc(rm * cm * sizeof(float));
	B = (float*)malloc(rm * cm * sizeof(float));
	C = (float*)malloc(rm * cm * sizeof(float));
	C_D = (float*)malloc(rm * cm * sizeof(float));

	for(int i=0;i<rm * cm; i++){
		A[i] = rand() % 100;
		B[i] = rand() % 20;
		C[i] = 0.0f;
	}
	Timer timer;

	startTimer(&timer);
	mmCpu(A, B, C, rm, cm);
	stopTimer(&timer);
	printElapsedTime(timer, "CPU mm time: ");
	
	startTimer(&timer);
	mmGpu(A, B, C_D, rm, cm);
	stopTimer(&timer);
	printElapsedTime(timer, "GPU mm total time: ");
	
	checkAB(C, C_D, rm * cm);
	free(A);
	free(B);
	free(C);
	free(C_D);
	
	return 0;
}