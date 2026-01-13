// Update package name if needed
package org.example;

import jcuda.*;
import jcuda.runtime.*;
public class Welcome
{
    public static void main(String args[])
    {
        Pointer pointer = new Pointer();
        JCuda.cudaMalloc(pointer, 4);
        System.out.println("Pointer: "+pointer);
        JCuda.cudaFree(pointer);
    }
}
