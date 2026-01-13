// Update package name if needed
package org.example;

import jcuda.*;
import jcuda.driver.*;

import static jcuda.driver.JCudaDriver.cuCtxSynchronize;

public class MatrixMultiplication {

    public static void main(String[] args) {
        int M = 1024, N = 1024, K = 1024;

        // Timing CPU execution
        float start = System.nanoTime();
        matrixMulCPU(M, N, K);
        float end = System.nanoTime();
        System.out.println("Time needed CPU: " + (int)(end - start));

        // Timing GPU execution
        start = System.nanoTime();
        matrixMulGPU(M, N, K);
        end = System.nanoTime();
        System.out.println("Time needed GPU: " + (int)(end - start));
    }

    public static void matrixMulCPU(int M, int N, int K) {
        // Allocate and fill the host input data
        float[] A = new float[M * K];
        float[] B = new float[K * N];
        float[] C = new float[M * N];

        // Initialize A and B with some values
        for (int i = 0; i < M * K; i++) A[i] = (float) (i % 100);
        for (int i = 0; i < K * N; i++) B[i] = (float) (i % 100);

        // Perform the matrix multiplication (C = A * B)
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                C[i * N + j] = 0;
                for (int k = 0; k < K; k++) {
                    C[i * N + j] += A[i * K + k] * B[k * N + j];
                }
            }
        }
    }


    public static void matrixMulGPU(int M, int N, int K) {
        // Allocate and fill the host input data
        float[] A = new float[M * K];
        float[] B = new float[K * N];
        float[] C = new float[M * N];

        // Initialize A and B with some values
        for (int i = 0; i < M * K; i++) A[i] = (float) (i % 100);
        for (int i = 0; i < K * N; i++) B[i] = (float) (i % 100);

        // Initialize the JCuda driver
        JCudaDriver.setExceptionsEnabled(true);
        CUdevice device = new CUdevice();
        CUcontext context = new CUcontext();
        CUmodule module = new CUmodule();
        CUfunction function = new CUfunction();

        // Initialize GPU
        JCudaDriver.cuInit(0);
        JCudaDriver.cuDeviceGet(device, 0);
        JCudaDriver.cuCtxCreate(context, 0, device);
      
        // Create .ptx from .cu with: nvcc -ptx kernel.cu -o kernel.ptx
        // Load the kernel from the .ptx file
        JCudaDriver.cuModuleLoad(module, "src/main/java/org/example/kernel.ptx");

        // Get the function handle for the kernel
        JCudaDriver.cuModuleGetFunction(function, module, "matrixMul");

        // Allocate device memory
        CUdeviceptr d_A = new CUdeviceptr();
        CUdeviceptr d_B = new CUdeviceptr();
        CUdeviceptr d_C = new CUdeviceptr();
        JCudaDriver.cuMemAlloc(d_A, A.length * Sizeof.FLOAT);
        JCudaDriver.cuMemAlloc(d_B, B.length * Sizeof.FLOAT);
        JCudaDriver.cuMemAlloc(d_C, C.length * Sizeof.FLOAT);

        // Copy matrices from host to device
        JCudaDriver.cuMemcpyHtoD(d_A, Pointer.to(A), A.length * Sizeof.FLOAT);
        JCudaDriver.cuMemcpyHtoD(d_B, Pointer.to(B), B.length * Sizeof.FLOAT);

        // Set up the kernel parameters
        Pointer kernelParameters = Pointer.to(
                Pointer.to(d_A),
                Pointer.to(d_B),
                Pointer.to(d_C),
                Pointer.to(new int[]{M}),
                Pointer.to(new int[]{N}),
                Pointer.to(new int[]{K})
        );

        // Define the grid and block dimensions
        int blockSizeX = 16;  // Adjust block size based on your GPU architecture
        int blockSizeY = 16;
        int gridSizeX = (int) Math.ceil((double) N / blockSizeX);
        int gridSizeY = (int) Math.ceil((double) M / blockSizeY);

        float start = System.nanoTime();

        // Launch the kernel
        JCudaDriver.cuLaunchKernel(function,
                gridSizeX, gridSizeY, 1,       // Grid size
                blockSizeX, blockSizeY, 1,     // Block size
                0, null, kernelParameters, null);

        cuCtxSynchronize();

        float end = System.nanoTime();
        System.out.println("Time needed for GPU Kernel execution: " + (int)(end - start));

        // Copy the result matrix C from device to host
        JCudaDriver.cuMemcpyDtoH(Pointer.to(C), d_C, C.length * Sizeof.FLOAT);


        // Clean up
        JCudaDriver.cuMemFree(d_A);
        JCudaDriver.cuMemFree(d_B);
        JCudaDriver.cuMemFree(d_C);
        JCudaDriver.cuCtxDestroy(context);
    }


}
