#define __CL_ENABLE_EXCEPTIONS

#include <fstream>
#include <iostream>
#include <iterator>
#include <stdio.h>
#include <numeric> 
#include <string>

#ifdef __APPLE__
#include <OpenCl/cl.hpp>
#else
#include <CL/cl.hpp>
#endif

#include <pcl/ocl/utils/impl/ocl_kernel_builder.hpp>
#ifndef __OCLManager__
#define __OCLManager__


using namespace std;

class OCLManager {
	public:
		OCLManager() {}
		~OCLManager(){}
		static OCLManager* getInstance();
		void destroyInstance();
		cl::Context getContext();
		cl::CommandQueue getQueue();
		cl::Program buildProgramFromSource(string& filename);
		cl::Program buildProgramFromSource(string& filename, string& fields);
		cl::Program buildProgramFromBinary(string& filename);
		void saveBinary(cl::Program* program, string& filename);
		cl::Device getDevice();
		
		private:
		vector<cl::Platform> platforms;
		vector<cl::Device> devices;
		cl::Context context;
		cl::CommandQueue queue;
		enum Cloud { XYZ, XYZRGBA};
		void initCL();
		void createDevicesAndContext();
		Cloud getCloudType(string& fields);
		static OCLManager* m_pInstance;
};
#endif

