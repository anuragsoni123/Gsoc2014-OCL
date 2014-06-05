typedef struct Poi
{
  float x;
  float y;
  float z;
  float padding;
}my_struct;

__kernel void remove_NAN_kernel(__global my_struct *input, __global int * indices, int size) 
{
   
      int id = get_global_id(0);

      if(isfinite (input[id].x) && isfinite (input[id].y) && isfinite (input[id].z))
      {
        indices[id] = 1;
      }
      
}

__kernel void passthrough(__global float *input, __global int * indices, int size, int index, float filter_limit_min_ , float filter_limit_max_, int negative) 
{

    int id = get_global_id(0);
    float val;
    negative =0;
    val = input[4*id+index];
    if(isfinite(val) && ((!negative && val > filter_limit_min_ && val < filter_limit_max_) || (negative && val <= filter_limit_min_ && val >= filter_limit_max_)))
     {
       indices[id] = 1;
     }
    //else
     //indices[id] = ceil(negative+0.0);
     
    
}
