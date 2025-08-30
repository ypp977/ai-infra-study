#include <stdio.h>

__global__ void hello_kernel() {
	printf("Hello from GPU (%d,%d,%d)\n",
		threadIdx.x,blockIdx.x,blockDim.x);
}

int main() {
	hello_kernel<<<2,3>>>();
	cudaDeviceSynchronize();
	return 0;
}
