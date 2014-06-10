typedef struct Poi
{
  float x;
  float y;
  float z;
  float val;
}my_struct;

__kernel void convolve_rows(__global my_struct *A, __local my_struct * data, __global my_struct *B, __constant float *kerneldata, int width, int height, int kernelSize, int BlockSize) 
{
   int idx = get_global_id(0);
   int idy = get_global_id(1);
   int idx_loc = get_local_id(0);
   int idy_loc = get_local_id(1);
   int half_width  = kernelSize/2;
   int comp = get_local_size(0);

   float4 temp1,temp2,temp3;
   temp1 = (float4)(0.0f,0.0f,0.0f,0.0f);
   temp2 = (float4)(0.0f,0.0f,0.0f,0.0f);
   temp3 = (float4)(0.0f,0.0f,0.0f,0.0f); 
  data[idy_loc*BlockSize+idx_loc+half_width] = A[idy*width+idx];
   
  // left Pixels
  if((idx_loc < half_width) && (idx>=half_width))
     data[idy_loc*BlockSize+idx_loc] = A[(idy)*width+idx-half_width];
 
 // Right Pixels
  if((idx_loc + half_width) >= comp && (idx < width-half_width))
    data[idy_loc*BlockSize+(idx_loc+kernelSize)] = A[(idy)*width+idx+half_width];
   idx_loc += half_width;
   barrier(CLK_LOCAL_MEM_FENCE);

   if(idx >= half_width && idx < width - half_width)
    {
     for(int i = 0; i <= kernelSize; i++)
       {
         temp2 = (float4)( kerneldata[i], kerneldata[i], kerneldata[i], kerneldata[i]);
         temp3 = (float4)(data[idy_loc*BlockSize+idx_loc-half_width+i].x, data[idy_loc*BlockSize+idx_loc-half_width+i].y,data[idy_loc*BlockSize+idx_loc-half_width+i].z,data[idy_loc*BlockSize+idx_loc-half_width+i].val);
         temp1 += temp2*temp3;
        
       }

    }
   B[idy*width+idx].x = temp1.x;
   B[idy*width+idx].y = temp1.y;
   B[idy*width+idx].z = temp1.z;
    
}

__kernel void convolve_cols(__global my_struct *A, __local my_struct * data, __global my_struct *B, __constant float *kerneldata, int width, int height, int kernelSize, int BlockSize) 
{
   int idx = get_global_id(0);
   int idy = get_global_id(1);
   int idx_loc = get_local_id(0);
   int idy_loc = get_local_id(1);

   int half_width  = kernelSize/2;
   int comp = get_local_size(1);
   float4 temp1,temp2,temp3;
   temp1 = (float4)(0.0f,0.0f,0.0f,0.0f);
   temp2 = (float4)(0.0f,0.0f,0.0f,0.0f);
   temp3 = (float4)(0.0f,0.0f,0.0f,0.0f);
 
  data[(idy_loc+half_width)*BlockSize+idx_loc] = A[idy*width+idx];
   // up Pixels
  if((idy_loc < half_width) && (idy>=half_width))
     data[idy_loc*BlockSize+idx_loc] = A[(idy-half_width)*width+idx];
 
 // Right Pixels
  if((idy_loc + half_width) >= comp && (idy < height-half_width))
    data[(idy_loc+kernelSize)*BlockSize+idx_loc] = A[(idy+half_width)*width+idx];
   idy_loc += half_width;
   barrier(CLK_LOCAL_MEM_FENCE);

   if(idy >= half_width && idy < height - half_width)
    {
     for(int i = 0; i <= kernelSize; i++)
       {
         temp2 = (float4)( kerneldata[i], kerneldata[i], kerneldata[i], kerneldata[i]);
         temp3 = (float4)(data[(idy_loc-half_width+i)*BlockSize+idx_loc].x, data[(idy_loc-half_width+i)*BlockSize+idx_loc].y,data[(idy_loc-half_width+i)*BlockSize+idx_loc].z,data[(idy_loc-half_width+i)*BlockSize+idx_loc].val);
         temp1 += temp2*temp3;
        
       }

    }
   B[idy*width+idx].x = temp1.x;
   B[idy*width+idx].y = temp1.y;
   B[idy*width+idx].z = temp1.z;
    
}

