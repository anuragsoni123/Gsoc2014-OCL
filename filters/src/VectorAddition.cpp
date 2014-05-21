#include<stdio.h>
using namespace std;
#include<pcl/ocl/utils/ocl_manager.h>
#include<CL/cl.hpp>
using namespace cl;
#include <utility>
#include <iostream>
#include <fstream>
#include <string>

void vector_add(int * A, int * B, int * C, const int LIST_SIZE)
{


    /*const int LIST_SIZE = 1000;
    int *A = new int[LIST_SIZE]; 
    int *B = new int[LIST_SIZE];
    int *C = new int[LIST_SIZE];
    for(int i = 0; i < LIST_SIZE; i++) {
        A[i] = i;
        B[i] = LIST_SIZE - i;
	//C[i]=0;
    }*/
printf("\n Calling inside Function");
OCLManager test,*test1;
test1 = test.getInstance();

Context context = test1->getContext();
CommandQueue queue = test1->getQueue();

// Read source file
//std::ifstream sourceFile("vector_add_kernel.cl");
//std::string sourceCode(std::istreambuf_iterator<char>(sourceFile),(std::istreambuf_iterator<char>()));
//Program::Sources source(1, std::make_pair(sourceCode.c_str(), sourceCode.length()+1));
//std::string sourceFile("vector_add_kernel.clbin");
//Program program = test1->buildProgramFromBinary(sourceFile);


//std::string binary("vector_add_kernel.clbin");
//test1->saveBinary(&program, binary);
std::string sourceFile("vector_add_kernel.cl");
Program program = test1->buildProgramFromSource(sourceFile);

// Make kernel
cl_int ret;
Kernel kernel(program, "vector_add",&ret);
if(ret != CL_SUCCESS)
       printf("\n kernel Build Error");
 // Create memory buffers
        Buffer bufferA = Buffer(context, CL_MEM_READ_ONLY, LIST_SIZE * sizeof(int));
        Buffer bufferB = Buffer(context, CL_MEM_READ_ONLY, LIST_SIZE * sizeof(int));
        Buffer bufferC = Buffer(context, CL_MEM_WRITE_ONLY, LIST_SIZE * sizeof(int));
 
        // Copy lists A and B to the memory buffers
        queue.enqueueWriteBuffer(bufferA, CL_TRUE, 0, LIST_SIZE * sizeof(int), A);
        queue.enqueueWriteBuffer(bufferB, CL_TRUE, 0, LIST_SIZE * sizeof(int), B);
 
        // Set arguments to kernel
        kernel.setArg(0, bufferA);
        kernel.setArg(1, bufferB);
        kernel.setArg(2, bufferC);

// Run the kernel on specific ND range
        NDRange global(LIST_SIZE);
        NDRange local(1);
        queue.enqueueNDRangeKernel(kernel, NullRange, global, local);
 
        // Read buffer C into a local list
        
        queue.enqueueReadBuffer(bufferC, CL_TRUE, 0, LIST_SIZE * sizeof(int), C);
 
        for(int i = 0; i < LIST_SIZE; i ++)
             std::cout << A[i] << " + " << B[i] << " = " << C[i] << std::endl; 

}
