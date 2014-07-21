#define __CL_ENABLE_EXCEPTIONS

#ifdef MAC
#include <OpenCL/cl.hpp>
#else
#include <CL/cl.hpp>
#endif

#include<pcl/ocl/utils/ocl_manager.h>

#define PROFILE

//DEFINE DIRECTION OF THE BITONIC SORT
/* Ascending: 0, Descending: -1 */
#define DIRECTION 0

void  bitonic_sort(int* array, int * indices, size_t length)
{
	std::string PROG_BSORT = "bitonic_sorting.cl";
	const char *KERNEL_BSORT_INIT = "bsort_init";
	const char *KERNEL_BSORT_STAGE_0 = "bsort_stage_0";
	const char *KERNEL_BSORT_STAGE_N = "bsort_stage_n";
	const char *KERNEL_BSORT_MERGE = "bsort_merge";
	const char *KERNEL_BSORT_MERGE_LAST = "bsort_merge_last";
		
  
	OCLManager* ocl = OCLManager::getInstance();

	cl::Program prg_bsort;
	prg_bsort = ocl->buildProgramFromSource(PROG_BSORT);
	
	cl::Kernel krn_bs_init(prg_bsort, KERNEL_BSORT_INIT);
	cl::Kernel krn_bs_s0(prg_bsort, KERNEL_BSORT_STAGE_0);
	cl::Kernel krn_bs_sn(prg_bsort, KERNEL_BSORT_STAGE_N);
	cl::Kernel krn_bs_mrg(prg_bsort, KERNEL_BSORT_MERGE);
	cl::Kernel krn_bs_mrg_last(prg_bsort, KERNEL_BSORT_MERGE_LAST);
	
		
	cl::Buffer buf_array(ocl->getContext(), CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR | CL_MEM_COPY_HOST_PTR, sizeof(int) * length, &array[0]);
	cl::Buffer buf_indices(ocl->getContext(), CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR | CL_MEM_COPY_HOST_PTR, sizeof(int) * length, &indices[0]);
	
	int krn_args=0;
			
	krn_bs_init.setArg(krn_args, buf_array);
	krn_bs_s0.setArg(krn_args, buf_array);
	krn_bs_sn.setArg(krn_args, buf_array);
	krn_bs_mrg.setArg(krn_args, buf_array);
	krn_bs_mrg_last.setArg(krn_args++, buf_array);
	
	krn_bs_init.setArg(krn_args, buf_indices);
	krn_bs_s0.setArg(krn_args, buf_indices);
	krn_bs_sn.setArg(krn_args, buf_indices);
	krn_bs_mrg.setArg(krn_args, buf_indices);
	krn_bs_mrg_last.setArg(krn_args++, buf_indices);
	
	int local_size = krn_bs_init.getWorkGroupInfo<CL_KERNEL_WORK_GROUP_SIZE>(ocl->getDevice());
        local_size = 512;
	cl::LocalSpaceArg l_arg_sort = cl::__local(8 * local_size * sizeof(cl_int));
        cl::LocalSpaceArg l_arg_ind = cl::__local(8 * local_size * sizeof(cl_int));

	krn_bs_init.setArg(krn_args, l_arg_sort);
	krn_bs_s0.setArg(krn_args, l_arg_sort);
	krn_bs_sn.setArg(krn_args, l_arg_sort);
	krn_bs_mrg.setArg(krn_args, l_arg_sort);
	krn_bs_mrg_last.setArg(krn_args++, l_arg_sort);

	krn_bs_init.setArg(krn_args, l_arg_ind);
	krn_bs_s0.setArg(krn_args, l_arg_ind);
	krn_bs_sn.setArg(krn_args, l_arg_ind);
	krn_bs_mrg.setArg(krn_args, l_arg_ind);
	krn_bs_mrg_last.setArg(krn_args++, l_arg_ind);

	size_t global_sort_size = length/8;
	if (global_sort_size < local_size) {
		local_size = global_sort_size;
	}
	

	ocl->getQueue().enqueueNDRangeKernel(krn_bs_init, 1, global_sort_size, local_size, NULL);
	

	size_t num_stages = global_sort_size/local_size;
	int stage, high_stage;
	/* Execute further stages */
	for (high_stage = 2; high_stage < num_stages; high_stage <<= 1) {

		krn_bs_s0.setArg(4, high_stage);
		krn_bs_sn.setArg(5, high_stage);

		for (stage = high_stage; stage > 1; stage >>= 1) {

			krn_bs_sn.setArg(4, stage);
			ocl->getQueue().enqueueNDRangeKernel(krn_bs_sn, 1, global_sort_size, local_size, NULL);

		}

		ocl->getQueue().enqueueNDRangeKernel(krn_bs_s0, 1, global_sort_size, local_size, NULL);

	}
	cl_int direction= DIRECTION;
	krn_bs_mrg.setArg(5, direction);
	krn_bs_mrg_last.setArg(4, direction);

	/* Perform the bitonic merge */
	for (stage = num_stages; stage > 1; stage >>= 1) {

		krn_bs_mrg.setArg(4, stage);

		ocl->getQueue().enqueueNDRangeKernel(krn_bs_mrg, 1, global_sort_size, local_size, NULL);		
	}
	ocl->getQueue().enqueueNDRangeKernel(krn_bs_mrg_last, 1, global_sort_size, local_size, NULL);

	void* mapped_array = ocl->getQueue().enqueueMapBuffer(buf_array, CL_TRUE, CL_MAP_READ
			| CL_MAP_WRITE, 0, sizeof(int) * length, NULL, NULL , NULL);

	memcpy(&array[0], mapped_array, length	* sizeof(int));

	ocl->getQueue().enqueueUnmapMemObject(buf_array, mapped_array);
	
	void* mapped_indices = ocl->getQueue().enqueueMapBuffer(buf_indices, CL_TRUE, CL_MAP_READ
			| CL_MAP_WRITE, 0, sizeof(int) * length, NULL, NULL, NULL);

	memcpy(&indices[0], mapped_indices, length * sizeof(int));

	ocl->getQueue().enqueueUnmapMemObject(buf_indices, mapped_indices);
	 
}
