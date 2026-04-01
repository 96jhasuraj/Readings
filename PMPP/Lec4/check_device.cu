#include <stdio.h>

int main() {

    int count = 0;
    cudaGetDeviceCount(&count);
    printf("Devices: %d\n", count);

    for (int i = 0; i < count; ++i) {
        cudaDeviceProp devProp;
        cudaGetDeviceProperties(&devProp, i);
        printf("\nDevice %d: %s\n", i, devProp.name);
        printf("\nmaxThreadsPerBlock %d\n", devProp.maxThreadsPerBlock);
        printf("\nmultiProcessorCount %d\n", devProp.multiProcessorCount);
        printf("\ntotalGlobalMem %zu\n", devProp.totalGlobalMem);
        printf("\nmaxThreadsDim %d %d %d\n", devProp.maxThreadsDim[0],devProp.maxThreadsDim[1],devProp.maxThreadsDim[2]);
        printf("\nmaxGridSize %d %d %d\n", devProp.maxGridSize[0], devProp.maxGridSize[1], devProp.maxGridSize[2]);
        printf("\nwarpsize %d\n", devProp.warpSize);
        printf("\nregsPerBlock %d\n", devProp.regsPerBlock);

    }
}