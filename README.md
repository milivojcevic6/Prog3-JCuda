# Prog3 JCuda Examples

This repository contains **basic [JCuda](https://javagl.de/jcuda.org/) examples** created while following a tutorial to understand how **CUDA-enabled GPUs can be used from Java**. The examples progress from a simple sanity check to real GPU computation and performance comparison with CPU implementations.

The goal of this repo is to:

* Verify that **JCuda is correctly installed and runnable**
* Use **CUDA libraries (JCurand)** from Java
* Write and execute a **custom CUDA kernel** from Java
* Compare **CPU vs GPU performance**, including the effect of memory transfers

---

## Prerequisites

* NVIDIA GPU with CUDA support
* [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) installed (`nvcc` available in terminal)
* Java (JDK 8+ recommended)
* JCuda libraries properly set up

---

## 1. JCuda Setup Test – `Welcome.java`

This is a **minimal sanity check** to verify that:

* JCuda is correctly linked
* CUDA runtime is accessible
* GPU memory allocation works

### What it does

* Allocates 4 bytes on the GPU using `cudaMalloc`
* Prints the pointer address
* Frees the allocated memory

If this program runs without errors, **JCuda is working correctly**.

---

## 2. Random Array Generation – `RandomArray.java`

This example demonstrates how to use the **[JCurand](https://javagl.de/jcuda.org/jcuda/jcurand/JCurand.html)** library to generate random numbers on the GPU and compare performance with a CPU implementation.

### What it does

* Generates an array of `n` random float values
* **CPU version**: Sequential random number generation
* **GPU version**: Random number generation using `JCurand`
* Measures and compares execution time for both approaches

### Key Concepts

* Using **CUDA libraries from Java**
* GPU-based random number generation
* Performance comparison between:

  * CPU (sequential)
  * GPU (parallel via JCuda)

This example shows that GPUs are not always the best choice: for small or trivial computations, the overhead of GPU execution can reduce the overall performance benefit.

---

## 3. Matrix Multiplication with Custom Kernel

Files involved:

* `MatrixMultiplication.java`
* `kernel.cu`

This is the **most advanced example** in the repository and demonstrates how to:

* Write a custom CUDA kernel
* Compile it to PTX
* Launch it from Java using JCuda
* Compare CPU vs GPU performance

### What it does

* Performs matrix multiplication: **C = A × B**
* Implements:

  * CPU version (sequential)
  * GPU version (CUDA kernel)
* Measures execution time under different conditions

### Performance Observations

* For large matrices (e.g. `n = m = k = 1024`):

  * **GPU is ~10× faster** when including memory allocation and transfers
  * **GPU is ~1000× faster** when comparing only:

    * Kernel execution + synchronization
    * CPU computation time

This clearly demonstrates that:

* GPU kernels are extremely fast
* Memory allocation and transfer overhead is significant
* Performance gains are highest for **large workloads**

---

## CUDA Kernel Compilation

The CUDA kernel is written in `kernel.cu` and must be compiled to **PTX** before running the Java code.

### Compile command

```bash
nvcc -ptx kernel.cu -o kernel.ptx
```

The generated `kernel.ptx` file is then loaded by `MatrixMultiplication.java` at runtime.

---

## Notes

* Performance depends heavily on GPU model and system configuration
* Results will vary for small input sizes due to overhead
* These examples are intended for **educational purposes**
