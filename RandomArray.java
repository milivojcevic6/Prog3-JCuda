// Update package name if needed
package org.example;

import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.jcurand.curandGenerator;

import java.util.Arrays;

import static jcuda.jcurand.JCurand.*;
import static jcuda.jcurand.curandRngType.CURAND_RNG_PSEUDO_DEFAULT;
import static jcuda.runtime.JCuda.cudaMalloc;
import static jcuda.runtime.JCuda.cudaMemcpy;
import static jcuda.runtime.cudaMemcpyKind.cudaMemcpyDeviceToHost;

public class RandomArray {

    public static void main(String[] args) {

        int n = 100000000;

        float startTime = System.nanoTime();
        generateFloatArrayCPU(n, false);

        float endTime = System.nanoTime();
        System.out.println("Time need for CPU: " + (int)(endTime - startTime) );


        startTime = System.nanoTime();
        generateFloatArrayGPU(n, false);           // ctrl + d
        endTime = System.nanoTime();
        System.out.println("Time need for GPU: " + (int)(endTime - startTime) );


    }

    private static void generateFloatArrayGPU(int n, boolean print) {

        // Allocate device memory
        Pointer deviceData = new Pointer();
        cudaMalloc(deviceData, n * Sizeof.FLOAT);

        // Create and initialize a pseudo-random number generator
        curandGenerator generator = new curandGenerator();
        curandCreateGenerator(generator, CURAND_RNG_PSEUDO_DEFAULT);
        curandSetPseudoRandomGeneratorSeed(generator, 1234);

        // Generate random numbers
        curandGenerateUniform(generator, deviceData, n);

        float [] array = new float[n];

        cudaMemcpy(Pointer.to(array), deviceData, n * Sizeof.FLOAT, cudaMemcpyDeviceToHost);

        if (print) {
            System.out.println(Arrays.toString(array));
        }

    }



    private static void generateFloatArrayCPU(int n, boolean print) {

        float [] array = new float[n];

        for (int i = 0; i < n; i++) {
            array[i] = (float) Math.random();
        }

        if (print) {
            System.out.println(Arrays.toString(array));
        }

    }

}
