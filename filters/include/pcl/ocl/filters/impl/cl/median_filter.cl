
typedef struct Poi
{
  float x;
  float y;
  float z;
  float padding;
}my_struct;


__kernel void median_filter_kernel(__global my_struct *A, __global my_struct *B, int width, int height, int radius, float max_allowed_movement_, __local float * input) 
{
   
 int idx = get_global_id(0);
 int idy = get_global_id(1);
 int idx_loc = get_local_id(0);
 int idy_loc = get_local_id(1);
 
 int BlockSize = get_local_size(0)+radius-1;
 int comp = get_local_size(0);

 int half_radius = radius/2;
 float temp;
 int l,j,minpos,y_dev,x_dev;
 int count = 0;
 float vals[25];

//for(j = 0; j < BlockSize*BlockSize ; j++)
   //input[j] = NAN;
/*
 input[(idy_loc)*BlockSize+(idx_loc)] = NAN;
 input[(idy_loc)*BlockSize+(idx_loc+comp)%BlockSize] = NAN;
 input[((idy_loc+comp)%BlockSize)*BlockSize+(idx_loc)] = NAN;
 input[((idy_loc+comp)%BlockSize)*BlockSize+(idx_loc+comp)%BlockSize] = NAN;

 barrier(CLK_LOCAL_MEM_FENCE);  
 */
if(idx < width && idy < height)
 {

 input[(idy_loc+half_radius)*BlockSize+(idx_loc+half_radius)] = A[idy*width+idx].z;   
    
 //Top row threads copy the row above 
  if((idy_loc < half_radius) && (idy >= half_radius))
     input[(idy_loc)*BlockSize+idx_loc+half_radius] = A[(idy-half_radius)*width+idx].z;
     
  //Bottom row threads copy the row below
  if(((idy_loc + half_radius) >= comp) && (idy < height-half_radius))
     input[(idy_loc+radius-1)*BlockSize+idx_loc+half_radius] = A[(idy+half_radius)*width+(idx)].z;
   
  // left Pixels
  if((idx_loc < half_radius) && (idx>=half_radius))
     input[(idy_loc+half_radius)*BlockSize+(idx_loc)] = A[(idy)*width+idx-half_radius].z;
  
  //Top Left
  if((idy_loc < half_radius) && (idy>=half_radius)&& (idx_loc < half_radius) && (idx>=half_radius))
        input[idy_loc*BlockSize+idx_loc] = A[(idy-half_radius)*width+(idx-half_radius)].z;
  
  //Bottom left
  if((idy_loc+half_radius) >= comp && (idy < height-half_radius) && (idx_loc < half_radius) && (idx>=half_radius))   
        input[(idy_loc+radius-1)*BlockSize+idx_loc] = A[(idy+half_radius)*width+(idx-half_radius)].z;
  
  // Right Pixels
  if((idx_loc + half_radius) >= comp && (idx < width-half_radius))
    
    input[(idy_loc+half_radius)*BlockSize+(idx_loc+radius-1)] = A[(idy)*width+idx+half_radius].z;

   //Top Right
  if((idy_loc < half_radius) && (idy >= half_radius) && (idx_loc + half_radius) >= comp && (idx < width-half_radius))
     input[idy_loc*BlockSize+idx_loc+radius-1] = A[(idy-half_radius)*width+idx+half_radius].z;
   
   // Bottom Right
   if((idy_loc+half_radius) >= comp && (idy < height-half_radius) && (idx_loc + half_radius) >= comp && (idx < width-half_radius))
     input[(idy_loc+radius-1)*BlockSize+(idx_loc+radius-1)] = A[(idy+half_radius)*width+(idx+half_radius)].z;
  
  // Synchronize the read into LMEM 
   barrier(CLK_LOCAL_MEM_FENCE);  
   
  idy_loc += half_radius;
  idx_loc += half_radius;
 
  if(isfinite(input[idy_loc*BlockSize+idx_loc]))
   {
	count = 0;
	for (y_dev = -half_radius; y_dev <= half_radius; ++y_dev)
	    for (x_dev = -half_radius; x_dev <= half_radius; ++x_dev)
	       {
		if (isfinite(input[(idy_loc+y_dev)*BlockSize+(idx_loc+x_dev)]))
		  {
	 	    vals[count] = input[(idy_loc+y_dev)*BlockSize+(idx_loc+x_dev)];
	 	    count = count+1;
		          
		  }

	        }


       if(count!=0)
        {
       
        //partial_sort 
         for (j = 0; j < (count/2)+1 ; ++j)
         {
            //Find position of minimum element
            minpos = j;
            for (l = j + 1; l < count; ++l)
              if (vals[l] < vals[minpos])
               minpos = l;
            // Put found minimum element in its place
            temp = vals[j];
            vals[j] = vals[minpos];
            vals[minpos] = temp;
         }
   
    	  if (fabs(vals[count/2] - input[idy_loc*BlockSize+idx_loc]) < max_allowed_movement_)
      	      B[idy*width+idx].z = vals[count/2];
    	  else
      	      B[idy*width+idx].z = input[idy_loc*BlockSize+idx_loc] + max_allowed_movement_ * (vals[count/2] - input[idy_loc*BlockSize+idx_loc]) / fabs(vals[count/2] - input[idy_loc*BlockSize+idx_loc]);
                           
          }
      
    }
}	

}



/*
__kernel void median_filter_kernel(__global my_struct *A, __global my_struct *B, int width, int height, int radius, float max_allowed_movement_) 
{

 int BlockSize = get_local_size(0)+radius-1;
 //int BlockSize = 20;
 int half_radius = radius/2;  
 int idx = get_global_id(0);
 int idy = get_global_id(1);
 max_allowed_movement_  = 0;
 int comp = get_local_size(0)-1;
 //int comp =15;
 int idx_loc = get_local_id(0);
 int idy_loc = get_local_id(1);
 
 float temp;
 int l,j,minpos,y_dev,x_dev,x_start;
 int count = 0;
 float vals[25];

 __local float input[144];
 //First Copy main data to Shared memory
  if(idx >=0 && idy >=0 && idx < width && idy < height)
    input[(idy_loc+half_radius)*BlockSize+(idx_loc+half_radius)] = (float)half_radius/10/*A[idy*width+idx].z*/;
  /*else 
    input[(idy_loc+half_radius)*BlockSize+(idx_loc+half_radius)] = NAN;
 barrier(CLK_LOCAL_MEM_FENCE);   

 //Top row threads copy the row above // can we include for previous also?
 /*  if((idy_loc - half_radius) < 0 && (idy >= half_radius))
     input[(idy_loc)*BlockSize+idx_loc+half_radius] = A[(idy-half_radius)*width+idx].z;
     
  //Bottom row threads copy the row below
   else if((idy_loc + half_radius) > comp && (idy <= height-half_radius))
     input[(idy_loc+radius)*BlockSize+idx_loc] = A[(idy+half_radius)*width+(idx)].z;
    
   // left Pixels
   else if((idx_loc - half_radius) < 0 && (idx >= half_radius))
     input[(idy_loc+half_radius)*BlockSize+(idx_loc-half_radius)] = A[(idy)*width+idx-half_radius].z;
     
   
    // Right Pixels
   else if((idx_loc + half_radius) > comp && (idx <= width-half_radius))
     input[(idy_loc+half_radius)*BlockSize+(idx_loc+radius)] = A[(idy)*width+idx+half_radius].z;

   //Top Left
   else if((idx_loc-half_radius) < 0 && (idy_loc-half_radius) < 0 && (idx>=half_radius) && (idy>=half_radius))
     input[idy_loc*BlockSize+idx_loc] = A[(idy-half_radius)*width+(idx-half_radius)].z;

   //Top Right
   else if((idy_loc-half_radius) > 0 && (idy >= half_radius) && (idx_loc+half_radius) > comp && (idx <= width-half_radius))
     input[idy_loc*width+idx+radius] = A[(idy-half_radius)*width+idx+radius].z;
   
   //Bottom left
   else if((idx_loc-half_radius) < 0 && (idy_loc+half_radius) > comp && (idx >= half_radius) && (idy <= height-half_radius))   
    input[(idy_loc+radius)*BlockSize+idx_loc] = A[(idy+half_radius)*width+(idx-half_radius)].z;
   
   // Bottom Right
   else if((idy_loc+half_radius) > comp && (idy <= height-half_radius) && (idx_loc+half_radius) > comp && (idx <= width-half_radius))
     input[(idy_loc+radius)*BlockSize+(idx_loc+radius)] = A[(idy+half_radius)*width+(idx+half_radius)].z;

c   else
      input[(idy_loc+half_radius)*BlockSize+(idx_loc+half_radius)] = NAN; 
  // Synchronize the read into LMEM
  */ 
  

   /*idy_loc += half_radius;
   idx_loc += half_radius;
  
   if(isfinite(input[idy_loc*BlockSize+idx_loc]))
      {
        count = 0;
        for (y_dev = -half_radius; y_dev <= half_radius; ++y_dev)
          for (x_dev = -half_radius; x_dev <= half_radius; ++x_dev)
          {
            if (idx + x_dev >= 0 && idx + x_dev < width && idy + y_dev >= 0 && idy + y_dev < height && isfinite(input[(idy_loc+y_dev)*BlockSize+(idx_loc+x_dev)]))
               {
                 vals[count] = input[(idy_loc+y_dev)*BlockSize+(idx_loc+x_dev)];
                 //vals[count] = 0.5;
                 count = count+1;
                                  
               }
                
          }
       count =1;
	if(count!=0)
        {
          
        //partial_sort 
         for (j = 0; j < (count/2)+1 ; ++j)
         {
            //Find position of minimum element
            minpos = j;
            for (l = j + 1; l < count; ++l)
              if (vals[l] < vals[minpos])
               minpos = l;
            // Put found minimum element in its place
            temp = vals[j];
            vals[j] = vals[minpos];
            vals[minpos] = temp;
         } 
          if (fabs(vals[count/2] - input[idy_loc*BlockSize+idx_loc]) < max_allowed_movement_)
            B[idy*width+idx].z = vals[count/2];
         else
            B[idy*width+idx].z = input[idy_loc*BlockSize+idx_loc] + max_allowed_movement_ * (vals[count/2] - input[idy_loc*BlockSize+idx_loc]) / fabs(vals[count/2] - input[idy_loc*BlockSize+idx_loc]);
         barrier(CLK_LOCAL_MEM_FENCE);                   
      }  
    }
   

}
*/





//2D data
/*
__kernel void median_filter_kernel(__constant my_struct *A, __global my_struct *B, int width, int height, int window_size_, float max_allowed_movement_) 
{
   
 int idy = get_global_id(0);
 int idx = get_global_id(1);
 float temp;
 int l,j,minpos,y_dev,x_dev;
 int count = 0;
 float vals[25];
 
     if(isfinite(A[idy*width+idx].x)&&isfinite(A[idy*width+idx].y)&&isfinite(A[idy*width+idx].z))
      {
        count = 0;
        for (y_dev = -window_size_/2; y_dev <= window_size_/2; ++y_dev)
          for (x_dev = -window_size_/2; x_dev <= window_size_/2; ++x_dev)
          {
            if (idx + x_dev >= 0 && idx + x_dev < width && idy + y_dev >= 0 && idy + y_dev < height && isfinite(A[(idy+y_dev)*width+(idx+x_dev)].x) && isfinite(A[(idy+y_dev)*width+(idx+x_dev)].y) && isfinite(A[(idy+y_dev)*width+(idx+x_dev)].z))
               {
                 vals[count] = A[(idy+y_dev)*width+(idx+x_dev)].z;
                 count = count+1;
                                  
               }
                
          }

       if(count!=0)
        {
       
        //partial_sort 
         for (j = 0; j < (count/2)+1 ; ++j)
         {
            //Find position of minimum element
            minpos = j;
            for (l = j + 1; l < count; ++l)
              if (vals[l] < vals[minpos])
               minpos = l;
            // Put found minimum element in its place
            temp = vals[j];
            vals[j] = vals[minpos];
            vals[minpos] = temp;
         }

          if (fabs(vals[count/2] - A[idy*width+idx].z) < max_allowed_movement_)
            B[idy*width+idx].z = vals[count/2];
         else
            B[idy*width+idx].z = A[idy*width+idx].z + max_allowed_movement_ * (vals[count/2] - A[idy*width+idx].z) / fabs(vals[count/2] - A[idy*width+idx].z);
                           
      }
      
    }
}
*/

/*__kernel void median_filter_kernel(__global my_struct *A, __global my_struct *B, int width, int height, int window_size_, float max_allowed_movement_) 
{
   
 int id = get_global_id(0);
 float temp;
 int l,j,minpos,y_dev,x_dev,y_start;
 int count = 0;
 float vals[25];
 
    for (int y_start = 0; y_start < width; ++y_start)
     if(isfinite(A[y_start*width+id].x))
      {
        count = 0;
        for (int y_dev = -window_size_/2; y_dev <= window_size_/2; ++y_dev)
          for (int x_dev = -window_size_/2; x_dev <= window_size_/2; ++x_dev)
          {
            if (id + x_dev >= 0 && id + x_dev < width && y_start + y_dev >= 0 && y_start + y_dev < height && isfinite(A[(y_start+y_dev)*width+(id+x_dev)].x) )
               {
                 vals[count] = A[(y_start+y_dev)*width+(id+x_dev)].z;
                 count = count+1;
                                  
               }
                
          }

       if(count!=0)
        {

         for (j = 0; j < (count/2)+1 ; ++j)
         {
            //Find position of minimum element
            minpos = j;
            for (l = j + 1; l < count; ++l)
              if (vals[l] < vals[minpos])
               minpos = l;
            // Put found minimum element in its place
            temp = vals[j];
            vals[j] = vals[minpos];
            vals[minpos] = temp;
         }


        if (fabs(vals[count/2] - A[y_start*width+id].z) < max_allowed_movement_)
            B[y_start*width+id].z = vals[count/2];
         else
            B[y_start*width+id].z = A[y_start*width+id].z + max_allowed_movement_ * (vals[count/2] - A[y_start*width+id].z) / fabs(vals[count/2] - A[y_start*width+id].z);
                           
      }
   //barrier(CLK_LOCAL_MEM_FENCE);
      
    }
}*/
