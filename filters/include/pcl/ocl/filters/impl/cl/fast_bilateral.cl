typedef struct Poi
{
  float x;
  float y;
  float z;
  float padding;
}my_struct;

__kernel void reduction_cloud(__global my_struct *A, __local float * inputmin, __local float * inputmax, __global float * maxi, __global float *mini) 
{
   int idx = get_global_id(0);
   int lid = get_local_id(0);
   int group_size = get_local_size(0);

   inputmin[lid] = A[idx].z;
   inputmax[lid] = inputmin[lid];
   barrier(CLK_LOCAL_MEM_FENCE);

   for(int i = group_size/2; i>0; i >>= 1) {
      if(lid < i) {
         inputmin[lid] = min(inputmin[lid], inputmin[lid+i]);
         inputmax[lid] = max(inputmax[lid], inputmax[lid+i]);	
      }
      barrier(CLK_LOCAL_MEM_FENCE);
   }

   if(lid == 0) {
      mini[get_group_id(0)] = inputmin[0];
      maxi[get_group_id(0)] = inputmax[0];
   }


}

__kernel void reduction_nan_cloud(__global my_struct *A, __local float * inputmin, __local float * inputmax, __global float * maxi, __global float *mini) 
{
   int idx = get_global_id(0);
   int lid = get_local_id(0);
   int group_size = get_local_size(0);

   inputmin[lid] = A[idx].z;
   inputmax[lid] = inputmin[lid];
   barrier(CLK_LOCAL_MEM_FENCE);

   for(int i = group_size/2; i>0; i >>= 1) {
      if(lid < i) {
          if(isfinite(inputmin[lid])&& inputmin[lid+i])
	{
	  inputmin[lid] = min(inputmin[lid], inputmin[lid+i]);
          inputmax[lid] = max(inputmax[lid], inputmax[lid+i]);	
	} 
     }
      barrier(CLK_LOCAL_MEM_FENCE);
   }

   if(lid == 0) {
      mini[get_group_id(0)] = inputmin[0];
      maxi[get_group_id(0)] = inputmax[0];
   }


}

