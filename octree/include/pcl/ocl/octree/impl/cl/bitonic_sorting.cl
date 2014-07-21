/* Sort elements within a vector */
#define VECTOR_SORT(input, input_cpi, dir)                        \
   comp = input < shuffle(input, mask2) ^ dir;                    \
   input = shuffle(input, as_uint4(comp * 2 + add2));             \
   input_cpi = shuffle(input_cpi, as_uint4(comp * 2 + add2));     \
   comp = input < shuffle(input, mask1) ^ dir;                    \
   input = shuffle(input, as_uint4(comp + add1));                 \
   input_cpi = shuffle(input_cpi, as_uint4(comp + add1));         \

#define VECTOR_SWAP(input1, input1_cpi, input2, input2_cpi, dir)  \
   temp = input1;                                                 \
   temp_cpi = input1_cpi;                                         \
   comp = (input1 < input2 ^ dir) * 4 + add3;                     \
   input1 = shuffle2(input1, input2, as_uint4(comp));             \
   input1_cpi = shuffle2(input1_cpi, input2_cpi, as_uint4(comp)); \
   input2 = shuffle2(input2, temp, as_uint4(comp));               \
   input2_cpi = shuffle2(input2_cpi, temp_cpi, as_uint4(comp));   \

/* Perform initial sort */
__kernel void bsort_init(__global int4 *g_data_idx, __global int4 *g_data_cpi, __local int4 *l_data_idx, __local int4 *l_data_cpi) {

   int dir;
   uint id, global_start, size, stride;
   int4 input1, input2, temp;
   
   int4 input1_cpi, input2_cpi, temp_cpi;

   
   int4 comp;
   
   uint4 mask1 = (uint4)(1, 0, 3, 2);
   uint4 mask2 = (uint4)(2, 3, 0, 1);
   uint4 mask3 = (uint4)(3, 2, 1, 0);

   int4 add1 = (int4)(1, 1, 3, 3);
   int4 add2 = (int4)(2, 3, 2, 3);
   int4 add3 = (int4)(1, 2, 2, 3);

   id = get_local_id(0) * 2;
   global_start = get_group_id(0) * get_local_size(0) * 2 + id;

   input1 = g_data_idx[global_start]; 
   input2 = g_data_idx[global_start+1];
   
   input1_cpi = g_data_cpi[global_start]; 
   input2_cpi = g_data_cpi[global_start+1];

   /* Sort input 1 - ascending */
   comp = input1 < shuffle(input1, mask1);
   input1 = shuffle(input1, as_uint4(comp + add1));
   input1_cpi = shuffle(input1_cpi, as_uint4(comp + add1));
   
   comp = input1 < shuffle(input1, mask2);
   input1 = shuffle(input1, as_uint4(comp * 2 + add2));
   input1_cpi = shuffle(input1_cpi, as_uint4(comp * 2 + add2));
   
   comp = input1 < shuffle(input1, mask3);
   input1 = shuffle(input1, as_uint4(comp + add3));
   input1_cpi = shuffle(input1_cpi, as_uint4(comp + add3));

   /* Sort input 2 - descending */
   comp = input2 > shuffle(input2, mask1);
   input2 = shuffle(input2, as_uint4(comp + add1));
   input2_cpi = shuffle(input2_cpi, as_uint4(comp + add1));
   
   comp = input2 > shuffle(input2, mask2);
   input2 = shuffle(input2, as_uint4(comp * 2 + add2));
   input2_cpi = shuffle(input2_cpi, as_uint4(comp * 2 + add2));
   
   comp = input2 > shuffle(input2, mask3);
   input2 = shuffle(input2, as_uint4(comp + add3));
   input2_cpi = shuffle(input2_cpi, as_uint4(comp + add3));

   /* Swap corresponding elements of input 1 and 2 */
   add3 = (int4)(4, 5, 6, 7);
   dir = get_local_id(0) % 2 * -1;
   temp = input1;
   temp_cpi = input1_cpi;
   comp = (input1 < input2 ^ dir) * 4 + add3;
   input1 = shuffle2(input1, input2, as_uint4(comp));
   input1_cpi = shuffle2(input1_cpi, input2_cpi, as_uint4(comp));
   
   input2 = shuffle2(input2, temp, as_uint4(comp));
   input2_cpi = shuffle2(input2_cpi, temp_cpi, as_uint4(comp));

   /* Sort data and store in local memory */
   VECTOR_SORT(input1, input1_cpi, dir);
   VECTOR_SORT(input2, input2_cpi, dir);
   l_data_idx[id] = input1;
   l_data_idx[id+1] = input2;
   
   l_data_cpi[id] = input1_cpi;
   l_data_cpi[id+1] = input2_cpi;
   /* Create bitonic set */
   for(size = 2; size < get_local_size(0); size <<= 1) {
      dir = (get_local_id(0)/size & 1) * -1;

      for(stride = size; stride > 1; stride >>= 1) {
         barrier(CLK_LOCAL_MEM_FENCE);
         id = get_local_id(0) + (get_local_id(0)/stride)*stride;
         VECTOR_SWAP(l_data_idx[id], l_data_cpi[id], l_data_idx[id + stride], l_data_cpi[id + stride], dir)
      }

      barrier(CLK_LOCAL_MEM_FENCE);
      id = get_local_id(0) * 2;
      input1 = l_data_idx[id]; input2 = l_data_idx[id+1];
      input1_cpi = l_data_cpi[id]; input2_cpi = l_data_cpi[id+1];
      temp = input1;
      temp_cpi = input1_cpi;
      comp = (input1 < input2 ^ dir) * 4 + add3;
      input1 = shuffle2(input1, input2, as_uint4(comp));
      input1_cpi = shuffle2(input1_cpi, input2_cpi, as_uint4(comp));
      
      input2 = shuffle2(input2, temp, as_uint4(comp));
      input2_cpi = shuffle2(input2_cpi, temp_cpi, as_uint4(comp));
      VECTOR_SORT(input1, input1_cpi, dir);
      VECTOR_SORT(input2, input2_cpi, dir);
      l_data_idx[id] = input1;
      l_data_idx[id+1] = input2;
      l_data_cpi[id] = input1_cpi;
      l_data_cpi[id+1] = input2_cpi;
   }

   /* Perform bitonic merge */
   dir = (get_group_id(0) % 2) * -1;
   for(stride = get_local_size(0); stride > 1; stride >>= 1) {
      barrier(CLK_LOCAL_MEM_FENCE);
      id = get_local_id(0) + (get_local_id(0)/stride)*stride;
      VECTOR_SWAP(l_data_idx[id], l_data_cpi[id], l_data_idx[id + stride], l_data_cpi[id + stride], dir)
   }
   barrier(CLK_LOCAL_MEM_FENCE);

   /* Perform final sort */
   id = get_local_id(0) * 2;
   input1 = l_data_idx[id]; input2 = l_data_idx[id+1];
   input1_cpi = l_data_cpi[id]; input2_cpi = l_data_cpi[id+1];
   temp = input1;
   temp_cpi = input1_cpi;
   comp = (input1 < input2 ^ dir) * 4 + add3;
   input1 = shuffle2(input1, input2, as_uint4(comp));
   input1_cpi = shuffle2(input1_cpi, input2_cpi, as_uint4(comp));
   input2 = shuffle2(input2, temp, as_uint4(comp));
   input2_cpi = shuffle2(input2_cpi, temp_cpi, as_uint4(comp));
   VECTOR_SORT(input1, input1_cpi, dir);
   VECTOR_SORT(input2, input2_cpi, dir);
   g_data_idx[global_start] = input1;
   g_data_cpi[global_start] = input1_cpi;
   g_data_idx[global_start+1] = input2;
   g_data_cpi[global_start+1] = input2_cpi;
}

/* Perform lowest stage of the bitonic sort */
__kernel void bsort_stage_0(__global int4 *g_data_idx, __global int4 *g_data_cpi, __local int4 *l_data_idx, __local int4 *l_data_cpi, 
                            uint high_stage) {

   int dir;
   uint id, global_start, stride;
   int4 input1, input2, temp;
   int4 comp;
   
   int4 input1_cpi, input2_cpi, temp_cpi;


   uint4 mask1 = (uint4)(1, 0, 3, 2);
   uint4 mask2 = (uint4)(2, 3, 0, 1);
   uint4 mask3 = (uint4)(3, 2, 1, 0);

   int4 add1 = (int4)(1, 1, 3, 3);
   int4 add2 = (int4)(2, 3, 2, 3);
   int4 add3 = (int4)(4, 5, 6, 7);

   /* Determine data location in global memory */
   id = get_local_id(0);
   dir = (get_group_id(0)/high_stage & 1) * -1;
   global_start = get_group_id(0) * get_local_size(0) * 2 + id;

   /* Perform initial swap */
   input1 = g_data_idx[global_start];
   input2 = g_data_idx[global_start + get_local_size(0)];
   
   input1_cpi = g_data_cpi[global_start];
   input2_cpi = g_data_cpi[global_start + get_local_size(0)];
   comp = (input1 < input2 ^ dir) * 4 + add3;
   l_data_idx[id] = shuffle2(input1, input2, as_uint4(comp));
   l_data_idx[id + get_local_size(0)] = shuffle2(input2, input1, as_uint4(comp));
   
   l_data_cpi[id] = shuffle2(input1_cpi, input2_cpi, as_uint4(comp));
   l_data_cpi[id + get_local_size(0)] = shuffle2(input2_cpi, input1_cpi, as_uint4(comp));
   /* Perform bitonic merge */
   for(stride = get_local_size(0)/2; stride > 1; stride >>= 1) {
      barrier(CLK_LOCAL_MEM_FENCE);
      id = get_local_id(0) + (get_local_id(0)/stride)*stride;
      VECTOR_SWAP(l_data_idx[id], l_data_cpi[id], l_data_idx[id + stride], l_data_cpi[id + stride], dir)
   }
   barrier(CLK_LOCAL_MEM_FENCE);

   /* Perform final sort */
   id = get_local_id(0) * 2;
   input1 = l_data_idx[id]; input2 = l_data_idx[id+1];
   input1_cpi = l_data_cpi[id]; input2_cpi = l_data_cpi[id+1];
   temp = input1;
   temp_cpi = input1_cpi;
   comp = (input1 < input2 ^ dir) * 4 + add3;
   input1 = shuffle2(input1, input2, as_uint4(comp));
   input1_cpi = shuffle2(input1_cpi, input2_cpi, as_uint4(comp));
   
   input2 = shuffle2(input2, temp, as_uint4(comp));
   input2_cpi = shuffle2(input2_cpi, temp_cpi, as_uint4(comp));
   VECTOR_SORT(input1, input1_cpi, dir);
   VECTOR_SORT(input2, input2_cpi, dir);

   /* Store output in global memory */
   g_data_idx[global_start + get_local_id(0)] = input1;
   g_data_idx[global_start + get_local_id(0) + 1] = input2;
   g_data_cpi[global_start + get_local_id(0)] = input1_cpi;
   g_data_cpi[global_start + get_local_id(0) + 1] = input2_cpi;
}

/* Perform successive stages of the bitonic sort */
__kernel void bsort_stage_n(__global int4 *g_data_idx, __global int4 *g_data_cpi, __local int4 *l_data_idx, __local int4 *l_data_cpi, 
                            uint stage, uint high_stage) {

   int dir;
   int4 input1, input2;
   int4 input1_cpi, input2_cpi;
   int4 comp, add;
   uint global_start, global_offset;

   add = (int4)(4, 5, 6, 7);

   /* Determine location of data in global memory */
   dir = (get_group_id(0)/high_stage & 1) * -1;
   global_start = (get_group_id(0) + (get_group_id(0)/stage)*stage) *
                   get_local_size(0) + get_local_id(0);
   global_offset = stage * get_local_size(0);

   /* Perform swap */
   input1 = g_data_idx[global_start];
   input1_cpi = g_data_cpi[global_start];
   
   input2 = g_data_idx[global_start + global_offset];
   input2_cpi = g_data_cpi[global_start + global_offset];
   comp = (input1 < input2 ^ dir) * 4 + add;
   g_data_idx[global_start] = shuffle2(input1, input2, as_uint4(comp));
   g_data_cpi[global_start] = shuffle2(input1_cpi, input2_cpi, as_uint4(comp));
   
   g_data_idx[global_start + global_offset] = shuffle2(input2, input1, as_uint4(comp));
   g_data_cpi[global_start + global_offset] = shuffle2(input2_cpi, input1_cpi, as_uint4(comp));
}

/* Sort the bitonic set */
__kernel void bsort_merge( __global int4 *g_data_idx, __global int4 *g_data_cpi, __local int4 *l_data_idx, __local int4 *l_data_cpi, uint stage, int dir) {

   int4 input1, input2;
   int4 input1_cpi, input2_cpi;
   int4 comp, add;
   uint global_start, global_offset;

   add = (int4)(4, 5, 6, 7);

   /* Determine location of data in global memory */
   global_start = (get_group_id(0) + (get_group_id(0)/stage)*stage) *
                   get_local_size(0) + get_local_id(0);
   global_offset = stage * get_local_size(0);

   /* Perform swap */
   input1 = g_data_idx[global_start];
   input1_cpi = g_data_cpi[global_start];
   
   input2 = g_data_idx[global_start + global_offset];
   input2_cpi = g_data_cpi[global_start + global_offset];
   comp = (input1 < input2 ^ dir) * 4 + add;
   g_data_idx[global_start] = shuffle2(input1, input2, as_uint4(comp));
   g_data_cpi[global_start] = shuffle2(input1_cpi, input2_cpi, as_uint4(comp));
   g_data_idx[global_start + global_offset] = shuffle2(input2, input1, as_uint4(comp));
   g_data_cpi[global_start + global_offset] = shuffle2(input2_cpi, input1_cpi, as_uint4(comp));
}

/* Perform final step of the bitonic merge */
__kernel void bsort_merge_last(__global int4 *g_data_idx, __global int4 *g_data_cpi, __local int4 *l_data_idx, __local int4 *l_data_cpi, int dir) {

   uint id, global_start, stride;
   int4 input1, input2, temp;
   int4 comp;
   
   int4 input1_cpi, input2_cpi, temp_cpi;

   uint4 mask1 = (uint4)(1, 0, 3, 2);
   uint4 mask2 = (uint4)(2, 3, 0, 1);
   uint4 mask3 = (uint4)(3, 2, 1, 0);

   int4 add1 = (int4)(1, 1, 3, 3);
   int4 add2 = (int4)(2, 3, 2, 3);
   int4 add3 = (int4)(4, 5, 6, 7);

   /* Determine location of data in global memory */
   id = get_local_id(0);
   global_start = get_group_id(0) * get_local_size(0) * 2 + id;

   /* Perform initial swap */
   input1 = g_data_idx[global_start];
   input1_cpi = g_data_cpi[global_start];
   input2 = g_data_idx[global_start + get_local_size(0)];
   input2_cpi = g_data_cpi[global_start + get_local_size(0)];
   comp = (input1 < input2 ^ dir) * 4 + add3;
   l_data_idx[id] = shuffle2(input1, input2, as_uint4(comp));
   l_data_idx[id + get_local_size(0)] = shuffle2(input2, input1, as_uint4(comp));
   
   l_data_cpi[id] = shuffle2(input1_cpi, input2_cpi, as_uint4(comp));
   l_data_cpi[id + get_local_size(0)] = shuffle2(input2_cpi, input1_cpi, as_uint4(comp));
   /* Perform bitonic merge */
   for(stride = get_local_size(0)/2; stride > 1; stride >>= 1) {
      barrier(CLK_LOCAL_MEM_FENCE);
      id = get_local_id(0) + (get_local_id(0)/stride)*stride;
      VECTOR_SWAP(l_data_idx[id], l_data_cpi[id], l_data_idx[id + stride], l_data_cpi[id + stride], dir)
   }
   barrier(CLK_LOCAL_MEM_FENCE);

   /* Perform final sort */
   id = get_local_id(0) * 2;
   input1 = l_data_idx[id]; input2 = l_data_idx[id+1];
   input1_cpi = l_data_cpi[id]; input2_cpi = l_data_cpi[id+1];
   temp = input1;
   temp_cpi = input1_cpi;
   comp = (input1 < input2 ^ dir) * 4 + add3;
   input1 = shuffle2(input1, input2, as_uint4(comp));
   input1_cpi = shuffle2(input1_cpi, input2_cpi, as_uint4(comp));
   input2 = shuffle2(input2, temp, as_uint4(comp));
   input2_cpi = shuffle2(input2_cpi, temp_cpi, as_uint4(comp));
   VECTOR_SORT(input1, input1_cpi, dir);
   VECTOR_SORT(input2, input2_cpi, dir);

   /* Store the result to global memory */
   g_data_idx[global_start + get_local_id(0)] = input1;
   g_data_idx[global_start + get_local_id(0) + 1] = input2;
   g_data_cpi[global_start + get_local_id(0)] = input1_cpi;
   g_data_cpi[global_start + get_local_id(0) + 1] = input2_cpi;
}
