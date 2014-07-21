#include <pcl/point_types.h>
#include <cfloat>
#include <iostream>
#include <vector>
#include <pcl/ocl/utils/ocl_manager.h>
#ifdef MAC
#include <OpenCL/cl.hpp>
#else
#include <CL/cl.hpp>
#endif

#define PROFILE

typedef pcl::PointXYZ PointT;

void get_min_max(cl::Buffer& buf_cloud, cl_float3& min, cl_float3& max, size_t cloud_size, std::string& fields)
{		
	std::string PROG_MIN_MAX = "min_max_3d.cl";
	const char *KERNEL_MIN_MAX_3D_START = "min_max_reduction_start";
	const char *KERNEL_MIN_MAX_3D = "min_max_reduction";
	const char *KERNEL_MIN_MAX_3D_COMPLETE = "min_max_reduction_complete";
	
	OCLManager* ocl = OCLManager::getInstance();
  	
  	cl::Program prg_min_max;
	prg_min_max = ocl->buildProgramFromSource(PROG_MIN_MAX,fields);
	
	cl::Kernel krn_min_max_3d(prg_min_max, KERNEL_MIN_MAX_3D);
	cl::Kernel krn_min_max_3d_complete(prg_min_max, KERNEL_MIN_MAX_3D_COMPLETE);
	cl::Kernel krn_min_max_3d_start(prg_min_max, KERNEL_MIN_MAX_3D_START);
	
	void* mapped_min_solution, *mapped_max_solution, *mapped_min_partial, *mapped_max_partial;
	//cl::Event e_min_max_start, e_min_max_partial, e_min_max_finish;
	//cl::Event e_buf_min_partial, e_buf_max_partial, e_buf_min_result, e_buf_max_result;
	
	int local_size = krn_min_max_3d.getWorkGroupInfo<CL_KERNEL_WORK_GROUP_SIZE>(ocl->getDevice());
	unsigned int num_groups = cloud_size/local_size;

	//cl_int err;
		
	std::vector<cl_float3> min_pts(num_groups);
	std::vector<cl_float3> max_pts(num_groups);
	
	cl::Buffer buf_min_partial(ocl->getContext(), CL_MEM_READ_WRITE
			| CL_MEM_ALLOC_HOST_PTR | CL_MEM_COPY_HOST_PTR, sizeof(cl_float3) * num_groups, &min_pts[0]);
	cl::Buffer buf_max_partial(ocl->getContext(), CL_MEM_READ_WRITE
			| CL_MEM_ALLOC_HOST_PTR | CL_MEM_COPY_HOST_PTR, sizeof(cl_float3) * num_groups, &max_pts[0]);	

	cl::Buffer buf_min_result(ocl->getContext(), CL_MEM_READ_WRITE
			| CL_MEM_ALLOC_HOST_PTR | CL_MEM_COPY_HOST_PTR, sizeof(cl_float3), &min);
	cl::Buffer buf_max_result(ocl->getContext(), CL_MEM_READ_WRITE
			| CL_MEM_ALLOC_HOST_PTR | CL_MEM_COPY_HOST_PTR, sizeof(cl_float3), &max);
		
	cl::LocalSpaceArg l_arg = cl::__local(local_size * sizeof(cl_float3));
	
	int krn_args = 0;

	krn_min_max_3d_start.setArg(krn_args++, buf_cloud);
	krn_min_max_3d_start.setArg(krn_args++, l_arg);
	krn_min_max_3d_start.setArg(krn_args++, l_arg);
	krn_min_max_3d_start.setArg(krn_args++, buf_min_partial);
	krn_min_max_3d_start.setArg(krn_args++, buf_max_partial);
	
	krn_args = 0;
	
	krn_min_max_3d.setArg(krn_args++, buf_min_partial);
	krn_min_max_3d.setArg(krn_args++, buf_max_partial);
	krn_min_max_3d.setArg(krn_args++, l_arg);
	krn_min_max_3d.setArg(krn_args++, l_arg);
	
	krn_args = 0;
	
	krn_min_max_3d_complete.setArg(krn_args++, buf_min_partial);
	krn_min_max_3d_complete.setArg(krn_args++, buf_max_partial);
	krn_min_max_3d_complete.setArg(krn_args++, l_arg);
	krn_min_max_3d_complete.setArg(krn_args++, l_arg);
	krn_min_max_3d_complete.setArg(krn_args++, buf_min_result);
	krn_min_max_3d_complete.setArg(krn_args++, buf_max_result);
	
	
	mapped_min_partial = ocl->getQueue().enqueueMapBuffer(buf_min_partial, CL_TRUE, CL_MAP_READ
			| CL_MAP_WRITE, 0, sizeof(cl_float3) * num_groups, NULL, NULL , NULL);
	memcpy(mapped_min_partial, &min_pts[0], sizeof(cl_float3) * num_groups);
	ocl->getQueue().enqueueUnmapMemObject(buf_min_partial, mapped_min_partial);
	
	mapped_max_partial = ocl->getQueue().enqueueMapBuffer(buf_max_partial, CL_TRUE, CL_MAP_READ
			| CL_MAP_WRITE, 0, sizeof(cl_float3) * num_groups, NULL, NULL, NULL);
	memcpy(mapped_max_partial, &max_pts[0], sizeof(cl_float3) * num_groups);
	ocl->getQueue().enqueueUnmapMemObject(buf_max_partial, mapped_min_partial);

	ocl->getQueue().enqueueNDRangeKernel(krn_min_max_3d_start, 0, cloud_size, local_size, NULL, NULL);
	
	
	size_t global_min_max_size = cloud_size/local_size;
	
	
	while(local_size<global_min_max_size){
		ocl->getQueue().enqueueNDRangeKernel(krn_min_max_3d, 0, global_min_max_size, local_size, NULL, NULL);
		global_min_max_size = global_min_max_size/local_size;
	}

	
	ocl->getQueue().enqueueNDRangeKernel(krn_min_max_3d_complete, 0, global_min_max_size, local_size, NULL);
		
	
	mapped_min_solution = ocl->getQueue().enqueueMapBuffer(buf_min_result, CL_TRUE, CL_MAP_READ
			| CL_MAP_WRITE, 0, sizeof(cl_float3), NULL, NULL, NULL);
	memcpy(&min, mapped_min_solution, sizeof(cl_float3));
	ocl->getQueue().enqueueUnmapMemObject(buf_min_result, mapped_min_solution);
	
	
	mapped_max_solution = ocl->getQueue().enqueueMapBuffer(buf_max_result, CL_TRUE, CL_MAP_READ
			| CL_MAP_WRITE, 0, sizeof(cl_float3), NULL, NULL, NULL);
	memcpy(&max, mapped_max_solution, sizeof(cl_float3));
	ocl->getQueue().enqueueUnmapMemObject(buf_max_result, mapped_max_solution);
	
}
