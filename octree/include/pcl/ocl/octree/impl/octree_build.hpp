#include <cfloat>
#include <iostream>
#include <vector>
#include<pcl/ocl/utils/ocl_manager.h>

#ifdef MAC
#include <OpenCL/cl.hpp>
#else
#include <CL/cl.hpp>
#endif


void octree_build(int* morton_codes, int* octree_beg, int* octree_end, int* octree_nodes, int* octree_parent, int * octree_codes, size_t cloud_size, int num_points, std::string& fields)
  {	
	std::string PROG_OCTREE = "morton.cl";
	const char *KERNEL_BUILD = "octree_build";
	OCLManager* ocl = OCLManager::getInstance();
        octree_beg[0] = 0;
        octree_end[0] = cloud_size;
  	cl::Program prg_build;
	prg_build = ocl->buildProgramFromSource(PROG_OCTREE, fields);
	cl::Kernel krn_build(prg_build, KERNEL_BUILD);
	cl_int err;
	void* map_beg;
	void* map_end;
	void* map_nodes;
        void * map_codes;
            
	cl::Buffer buf_morton(ocl->getContext(), CL_MEM_READ_WRITE
	| CL_MEM_ALLOC_HOST_PTR | CL_MEM_COPY_HOST_PTR, sizeof(int) * cloud_size, &morton_codes[0]);
	
	cl::Buffer buf_beg(ocl->getContext(), CL_MEM_READ_WRITE
	| CL_MEM_ALLOC_HOST_PTR | CL_MEM_COPY_HOST_PTR, sizeof(int) * num_points, &octree_beg[0]);
 
        cl::Buffer buf_end(ocl->getContext(), CL_MEM_READ_WRITE
	| CL_MEM_ALLOC_HOST_PTR | CL_MEM_COPY_HOST_PTR, sizeof(int) * num_points, &octree_end[0]);
		
        cl::Buffer buf_nodes(ocl->getContext(), CL_MEM_READ_WRITE
	| CL_MEM_ALLOC_HOST_PTR | CL_MEM_COPY_HOST_PTR, sizeof(int) * num_points, &octree_nodes[0]);

        cl::Buffer buf_parent(ocl->getContext(), CL_MEM_READ_WRITE
	| CL_MEM_ALLOC_HOST_PTR | CL_MEM_COPY_HOST_PTR, sizeof(int) * num_points, &octree_parent[0]);
	
	cl::Buffer buf_codes(ocl->getContext(), CL_MEM_READ_WRITE
	| CL_MEM_ALLOC_HOST_PTR | CL_MEM_COPY_HOST_PTR, sizeof(int) * num_points, &octree_codes[0]);
	
	int krn_args = 0;
	krn_build.setArg(krn_args++, buf_morton);
	krn_build.setArg(krn_args++, buf_beg);
	krn_build.setArg(krn_args++, buf_end);
	krn_build.setArg(krn_args++, buf_nodes);
	krn_build.setArg(krn_args++, buf_parent);
        krn_build.setArg(krn_args++, buf_codes);
        cl::NDRange global(1024);        
	
	ocl->getQueue().enqueueNDRangeKernel(krn_build, 0, global, cl::NullRange, NULL, NULL);
	
        
	map_beg = ocl->getQueue().enqueueMapBuffer(buf_beg, CL_TRUE, CL_MAP_READ
			| CL_MAP_WRITE, 0, sizeof(int) * num_points, NULL, NULL, NULL);
	memcpy(&octree_beg[0], map_beg, sizeof(int) * num_points);
	ocl->getQueue().enqueueUnmapMemObject(buf_beg, map_beg);


	map_end = ocl->getQueue().enqueueMapBuffer(buf_end, CL_TRUE, CL_MAP_READ
			| CL_MAP_WRITE, 0, sizeof(int) * num_points, NULL, NULL, NULL);
	memcpy(&octree_end[0], map_end, sizeof(int) * num_points);
	ocl->getQueue().enqueueUnmapMemObject(buf_end, map_end);

	map_nodes = ocl->getQueue().enqueueMapBuffer(buf_nodes, CL_TRUE, CL_MAP_READ
			| CL_MAP_WRITE, 0, sizeof(int) * num_points, NULL, NULL, NULL);
	memcpy(&octree_nodes[0], map_nodes, sizeof(int) * num_points);
	ocl->getQueue().enqueueUnmapMemObject(buf_nodes, map_nodes);

	map_codes = ocl->getQueue().enqueueMapBuffer(buf_codes, CL_TRUE, CL_MAP_READ
			| CL_MAP_WRITE, 0, sizeof(int) * num_points, NULL, NULL, NULL);
	memcpy(&octree_codes[0], map_codes, sizeof(int) * num_points);
	ocl->getQueue().enqueueUnmapMemObject(buf_codes, map_codes);
}

 
