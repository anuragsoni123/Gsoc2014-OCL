#include <cfloat>
#include <iostream>
#include <vector>
#include<pcl/ocl/utils/ocl_manager.h>

#ifdef MAC
#include <OpenCL/cl.hpp>
#else
#include <CL/cl.hpp>
#endif

#define PROFILE

void calculate_morton(cl::Buffer& buf_cloud, size_t cloud_size, cl_float3& min, cl_float3& max, int* morton_codes, std::string& fields)
{
	std::string PROG_MORTON = "morton.cl";
	const char *KERNEL_MORTON = "calculate_morton";
	OCLManager* ocl = OCLManager::getInstance();
  
  	cl::Program prg_morton;
	prg_morton = ocl->buildProgramFromSource(PROG_MORTON, fields);
	cl::Kernel krn_morton(prg_morton, KERNEL_MORTON);
	void* map_morton;	
	
	cl::Buffer buf_morton(ocl->getContext(), CL_MEM_READ_WRITE
	| CL_MEM_ALLOC_HOST_PTR | CL_MEM_COPY_HOST_PTR, sizeof(int) * cloud_size, &morton_codes[0]);
	int krn_args = 0;
	krn_morton.setArg(krn_args++, buf_cloud);
	krn_morton.setArg(krn_args++, min);
	krn_morton.setArg(krn_args++, max);
	krn_morton.setArg(krn_args++, buf_morton);
		
	ocl->getQueue().enqueueNDRangeKernel(krn_morton, 0, cloud_size, cl::NullRange, NULL);
	
	map_morton = ocl->getQueue().enqueueMapBuffer(buf_morton, CL_TRUE, CL_MAP_READ
			| CL_MAP_WRITE, 0, sizeof(int) * cloud_size, NULL, NULL);
	memcpy(&morton_codes[0], map_morton, sizeof(int) * cloud_size);
	ocl->getQueue().enqueueUnmapMemObject(buf_morton, map_morton);

}
