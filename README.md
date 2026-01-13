👉 [Scroll up for English version](#jcuda-examples)


# JCuda Primeri

Ta repozitorij vsebuje **osnovne primere [JCuda](https://javagl.de/jcuda.org/)**, ustvarjene med sledenjem vadnici za razumevanje, kako lahko GPU-je s podporo za CUDA uporabljamo iz Jave. Primeri postopoma napredujejo od preprostega preverjanja delovanja do dejanskih izračunov na GPU-ju in primerjav zmogljivosti s CPU-jem.

Cilji tega repozitorija so:

* Preveriti, ali je **JCuda pravilno nameščena in delujoča**
* Uporabljati **CUDA knjižnice (JCurand)** iz Jave
* Write and execute a **custom CUDA kernel** from Java
* Napisati in zagnati **lastno CUDA jedro (kernel)** iz Jave
* Primerjati zmogljivost **CPU-ja in GPU-ja**, vključno z vplivom prenosa podatkov

---

## Predpogoji

* NVIDIA GPU s podporo za CUDA
* Nameščen [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) (`nvcc` mora biti na voljo v terminalu)
* Java (priporočeno JDK 8 ali novejši)
* Pravilno [nastavljene](https://javagl.de/jcuda.org/downloads/downloads.html) JCuda knjižnice

---

## 1. Preverjanje nastavitve JCuda – `Welcome.java`

To je **minimalni test**, s katerim preverimo, da:

* je JCuda pravilno povezana
* je CUDA runtime dostopen
* dodeljevanje pomnilnika na GPU-ju deluje

### Kaj program naredi

* Dodeli 4 bajte pomnilnika na GPU-ju z `cudaMalloc`
* Izpiše naslov kazalca (pointer)
* Sprosti dodeljeni pomnilnik

Če se program izvede brez napak, **JCuda deluje pravilno**.

---

## 2. Generiranje naključnega polja – `RandomArray.java`

Ta primer prikazuje uporabo knjižnice **[JCurand](https://javagl.de/jcuda.org/jcuda/jcurand/JCurand.html)** za generiranje naključnih števil na GPU-ju ter primerjavo zmogljivosti s CPU implementacijo.

### Kaj program naredi

* Ustvari polje `n` naključnih realnih števil (float)
* **CPU različica**: zaporedno generiranje naključnih števil
* **GPU različica**: generiranje naključnih števil z uporabo `JCurand`
* Izmeri in primerja čas izvajanja obeh pristopov

### Ključni koncepti

* Uporaba **CUDA knjižnic iz Jave**
* Generiranje naključnih števil na GPU-ju
* Primerjava zmogljivosti med:

  * CPU (zaporedno)
  * GPU (vzporedno prek JCuda)

Ta primer pokaže, da GPU ni vedno najboljša izbira: pri majhnih ali trivialnih izračunih lahko režijski stroški uporabe GPU-ja zmanjšajo ali izničijo prednosti v zmogljivosti.

---

## 3. Množenje matrik z lastnim jedrom

Vključene datoteke:

* `MatrixMultiplication.java`
* `kernel.cu`

To je **najnaprednejši primer** v repozitoriju, ki prikazuje, kako:

* napisati lastno CUDA jedro
* ga prevesti v PTX obliko
* ga zagnati iz Jave z uporabo JCuda
* primerjati zmogljivost CPU-ja in GPU-ja

### Kaj program naredi

* Izvede množenje matrik: **C = A × B**
* Implementira:

  * CPU različico (zaporedno)
  * GPU različico (CUDA jedro)
* Izmeri čas izvajanja v različnih pogojih

### Opazovanja glede zmogljivosti

* Pri velikih matrikah (npr. `n = m = k = 1024`):

  * **GPU je približno 10× hitrejši**, če upoštevamo tudi dodeljevanje in prenos pomnilnika
  * **GPU je približno 1000× hitrejši**, če primerjamo samo:

    * izvajanje jedra + sinhronizacijo
    * čas CPU izračuna

To jasno pokaže, da:

* so CUDA jedra izjemno hitra
* ma prenos podatkov in dodeljevanje pomnilnika velik vpliv
* so dobitki največji pri **velikih delovnih obremenitvah**

---

## Prevajanjem CUDA jedra

CUDA jedro je zapisano v datoteki `kernel.cu` in ga je pred zagonom Java kode treba prevesti v **PTX** obliko.

### Ukaz za prevajanje

```bash
nvcc -ptx kernel.cu -o kernel.ptx
```

Ustvarjena datoteka `kernel.ptx` se nato naloži v razredu `MatrixMultiplication.java` med izvajanjem.

---

## Opombe

* Zmogljivost je močno odvisna od modela GPU-ja in konfiguracije sistema
* Pri majhnih vhodnih podatkih se rezultati razlikujejo zaradi režijskih stroškov
* Primeri so namenjeni **izobraževalnim namenom**


# JCuda Examples

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
