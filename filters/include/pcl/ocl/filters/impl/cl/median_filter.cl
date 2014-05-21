
typedef struct Poi
{
  float x;
  float y;
  float z;
  float padding;
}my_struct;

__kernel void median_filter_kernel(__global my_struct *A, __global my_struct *B, int width, int height, int window_size_, float max_allowed_movement_) 
{
   
 int idy = get_global_id(0);
 int idx = get_global_id(1);
 //int window_size_ = 5;
 //int width = 640, height = 480; 
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
       //Order elements (only half of them)

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

