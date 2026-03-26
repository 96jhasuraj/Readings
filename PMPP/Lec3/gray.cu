#include "timer.h"

__global__
void grb2gray_kernel(unsigned char* red, unsigned char* green, unsigned char* blue, unsigned char* gray, int h, int w) {
	unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
	unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;

	if (row < h && col < w) {
		int idx = row * w + col;
		gray[idx] = 0.299f * red[idx] + 0.587f * green[idx] + 0.114f * blue[idx];
	}
}

void grb2gray_gpu(unsigned char *red, unsigned char *green, unsigned char *blue, unsigned char *gray, int h , int w) {
	Timer timer;
	startTimer(&timer);
	unsigned char* R, * G, * B, * GRAY;
	cudaMalloc((void**)&R, h * w * sizeof(unsigned char));
	cudaMalloc((void**)&G, h * w * sizeof(unsigned char));
	cudaMalloc((void**)&B, h * w * sizeof(unsigned char));
	cudaMalloc((void**)&GRAY, h * w * sizeof(unsigned char));
	cudaDeviceSynchronize();
	stopTimer(&timer);
	printElapsedTime(timer, "Memory allocation time: ");

	startTimer(&timer);
	cudaMemcpy(R, red, h * w * sizeof(unsigned char), cudaMemcpyHostToDevice);
	cudaMemcpy(G, green, h * w * sizeof(unsigned char), cudaMemcpyHostToDevice);
	cudaMemcpy(B, blue, h * w * sizeof(unsigned char), cudaMemcpyHostToDevice);
	cudaDeviceSynchronize();
	stopTimer(&timer);
	printElapsedTime(timer, "Memory copy to device time: ");

	startTimer(&timer);
	dim3 threadDimPerBlock(32, 32, 1);
	dim3 numBlocks((w+ threadDimPerBlock.x-1) / threadDimPerBlock.x, (h+ threadDimPerBlock.y-1) / threadDimPerBlock.y);
	grb2gray_kernel <<<numBlocks,threadDimPerBlock>>>(R, G, B, GRAY, h, w);
	cudaDeviceSynchronize();
	stopTimer(&timer);
	printElapsedTime(timer, "kernel time");
	cudaMemcpy(gray, GRAY, h * w * sizeof(unsigned char), cudaMemcpyDeviceToHost);
	cudaFree(R); 
	cudaFree(G); 
	cudaFree(B); 
	cudaFree(GRAY);
}

void save_gray_image(const char* filename, unsigned char* gray, int w, int h) {
	FILE* f = fopen(filename, "wb");
	fprintf(f, "P5\n%d %d\n255\n", w, h);
	fwrite(gray, sizeof(unsigned char), w * h, f);
	fclose(f);
}
void save_rgb_image(const char* filename, unsigned char* r, unsigned char* g, unsigned char* b, int w, int h) {
	FILE* f = fopen(filename, "wb");
	fprintf(f, "P6\n%d %d\n255\n", w, h);
	for (int i = 0; i < w * h; i++) {
		fputc(r[i], f);
		fputc(g[i], f);
		fputc(b[i], f);
	}
	fclose(f);
}

void grb2gray_cpu(unsigned char* red, unsigned char* green, unsigned char* blue, unsigned char* gray, int h, int w) {
	for(int i=0;i<w;i++) {
		for(int j=0;j<h;j++) {
			int idx = j * w + i;
			gray[idx] = 0.299f * red[idx] + 0.587f * green[idx] + 0.114f * blue[idx];
		}
	}
}
int main()
{
	int h = 1024, w = 1024;
	unsigned char* red = (unsigned char*)malloc(h * w * sizeof(unsigned char));
	unsigned char* green = (unsigned char*)malloc(h * w * sizeof(unsigned char));
	unsigned char* blue = (unsigned char*)malloc(h * w * sizeof(unsigned char));
	unsigned char* gray = (unsigned char*)malloc(h * w * sizeof(unsigned char));
	
	for (int row = 0; row < h; row++) {
		for (int col = 0; col < w; col++) {
			int idx = row * w + col;
			red[idx] = (col * 255) / w;
			green[idx] = (row * 255) / h;
			blue[idx] = ((row + col) * 255) / (w + h);
		}
	}
	
	save_rgb_image("input.pgm", red, green, blue, w, h);
	grb2gray_gpu(red, green, blue, gray, h, w);
	save_gray_image("output.pgm", gray, w, h);

	Timer timer;
	startTimer(&timer);
	grb2gray_cpu(red, green, blue, gray, h, w);
	stopTimer(&timer);
	printElapsedTime(timer, "CPU time: ");

	free(red);
	free(green);
	free(blue);
	free(gray);
}