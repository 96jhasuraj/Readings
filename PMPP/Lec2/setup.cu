#include <stdio.h>

__global__ void hello_kernel()
{
	printf("Hello from the kernel\n thread : %d blockDim: %d blockIdx :%d \n",threadIdx.x,blockDim.x, blockIdx.x);
}
int main()
{
	hello_kernel<< <2, 4 >> > ();
	cudaDeviceSynchronize();
	printf("Hello from the host\n");
	return 0;
}
