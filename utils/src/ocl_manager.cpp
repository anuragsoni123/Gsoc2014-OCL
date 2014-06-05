#define __CL_ENABLE_EXCEPTIONS

//#include "../include/pcl/ocl/utils/ocl_manager.h"
#include <pcl/ocl/utils/ocl_manager.h>

#ifdef __APPLE__
#include <OpenCl/cl.hpp>
#else
#include <CL/cl.hpp>
#endif

#include <fstream>
#include <iostream>
#include <iterator>
#include <stdio.h>
#include <numeric> 
using namespace std;



OCLManager* OCLManager::m_pInstance = NULL;
OCLManager* OCLManager::getInstance() {
     if(NULL == m_pInstance ) {
            m_pInstance = new OCLManager();
	    m_pInstance->initCL();
     }
     return m_pInstance;
}
void OCLManager::destroyInstance() {
     delete m_pInstance;
     m_pInstance = NULL;
}
void OCLManager::initCL() {
    	this->createDevicesAndContext();
        cl::CommandQueue q(context, devices[0], CL_QUEUE_PROFILING_ENABLE);
        queue = q;
}

#define GPU

void OCLManager::createDevicesAndContext()
{
	
	
   vector<cl::Device> gpu_devices, cpu_devices, acc_devices;
   std::string device_name;
   cl_uint i;
   try {
	   #ifdef GPU
       //Access all devices in first platform
         cl::Platform::get(&platforms);
         platforms[0].getDevices(CL_DEVICE_TYPE_GPU, &devices);
      //Create context and access device names
         cl::Context ctx_(devices);
         context = ctx_;
         gpu_devices = context.getInfo<CL_CONTEXT_DEVICES>();
         for(i=0; i<gpu_devices.size(); i++) {
            device_name = gpu_devices[i].getInfo<CL_DEVICE_NAME>();
            //cout << "Device: " << device_name.c_str() << endl;
         }
         #endif
      if(gpu_devices.size()==0){
		#ifdef CPU
         // Access all devices in first platform
         cl::Platform::get(&platforms);
         platforms[0].getDevices(CL_DEVICE_TYPE_CPU, &devices);
         
         // Create context and access device names
         cl::Context ctx_(devices);
         context = ctx_;
         cpu_devices = context.getInfo<CL_CONTEXT_DEVICES>();
         for(i=0; i<cpu_devices.size(); i++) {
            device_name = cpu_devices[i].getInfo<CL_DEVICE_NAME>();
            //cout << "Device: " << device_name.c_str() << endl;
         }
         #endif
      }
       if(cpu_devices.size()==0 && gpu_devices.size()==0){
		#ifdef ACC
            // Access all devices in first platform
            cl::Platform::get(&platforms);
            platforms[0].getDevices(CL_DEVICE_TYPE_ACCELERATOR, &devices);
            
         // Create context and access device names
            cl::Context ctx__(devices);
            context = ctx__;
         acc_devices = context.getInfo<CL_CONTEXT_DEVICES>();
         for(i=0; i<acc_devices.size(); i++) {
            device_name = acc_devices[i].getInfo<CL_DEVICE_NAME>();
            cout<<" "<<endl<<endl<<endl;
            //cout << "Device: " << device_name.c_str() << endl;
         }
         #endif
       }
         if(cpu_devices.size()==0 && gpu_devices.size()==0 && acc_devices.size()==0)
        	 cerr<<"No OpenCL device found!!"<<endl;
      }
      catch(cl::Error e) {
         cout << e.what() << ": Error code " << e.err() << endl;
      }
}
/*
size_t OCLManager::getDeviceLocalSize(){
	size_t local_size;
	cl_int err;
	err = clGetDeviceInfo(devices[0], CL_DEVICE_MAX_WORK_GROUP_SIZE,
			sizeof(local_size), &local_size, NULL);
	if(err < 0) {
		perror("Couldn't obtain device information");
		exit(1);
	}
	return local_size;
}
*/

cl::Context OCLManager::getContext(){
	return context;
}
cl::CommandQueue OCLManager::getQueue(){
	return queue;
}

 cl::Program OCLManager::buildProgramFromSource(std::string& filename) {
		std::ifstream programFile((char*) filename.c_str());
	 	std::string programString(std::istreambuf_iterator<char>(programFile),(std::istreambuf_iterator<char>()));
	 	cl::Program::Sources source(1, std::make_pair(programString.c_str(),programString.length()+1));
	 	cl::Program program(context, source);
	 	cl_int ret = program.build(devices);
                   
    return program;
}
 

 
 cl::Program OCLManager::buildProgramFromBinary(std::string& filename)
 {
	cl::Program::Binaries binary;
	cl::Program program;
	FILE *fp = fopen((char*) filename.c_str(), "rb");
	size_t binarySize, read;
	fseek(fp, 0, SEEK_END);
	binarySize = ftell(fp);
	rewind(fp);
	unsigned char *programBinary = new unsigned char[binarySize];
	
	read = fread(programBinary, 1, binarySize, fp);
	if(read==0)
		throw cl::Error(-999, "Empty binary!");
	binary.insert(binary.begin(), std::make_pair(programBinary, binarySize));
	program = cl::Program(context, devices, binary);
	program.build(devices);
	fclose(fp);
	return program;
  }

 
 void OCLManager::saveBinary(cl::Program* program, std::string& filename){
 // Allocate some memory for all the kernel binary data
 const std::vector<unsigned long> binSizes = program->getInfo<CL_PROGRAM_BINARY_SIZES>();    
 std::vector<char> binData (std::accumulate(binSizes.begin(),binSizes.end(),0));    
 char* binChunk = &binData[0] ;  
 //A list of pointers to the binary data    
 std::vector<char*> binaries;    
 for(unsigned int i = 0; i<binSizes.size(); ++i) {    
      binaries.push_back(binChunk) ;
      binChunk += binSizes[i] ;    
 }    
 std::cout<<"Program name: " << (char*) filename.c_str() << std::endl; 
 program->getInfo(CL_PROGRAM_BINARIES , &binaries[0] ) ;    
 std::ofstream binaryfile((char*) filename.c_str(), std::ios::binary);    
 for (unsigned int i = 0; i < binaries.size(); ++i)    
      binaryfile.write(binaries[i], binSizes[i]); 

 }
//cl::Device getdevice() ;
cl::Device OCLManager::getdevice()
{

 return devices[0];
}
