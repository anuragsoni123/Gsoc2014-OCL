typedef struct {
	union {
		float data[4];
		struct {
			float x;
			float y;
			float z;
			float w;
		};
	};
} PointXYZ;

typedef struct {
	union {
		float data[4];
		float rgba[4];
		struct {
			float x;
			float y;
			float z;
			float w;
		};
		struct {
			float r;
			float g;
			float b;
			float a;
		};
	};
} PointXYZRGBA;


float3 max_pt(float3 ac_max, float3 element) {
	ac_max.x = fmax(ac_max.x, element.x);
	ac_max.y = fmax(ac_max.y, element.y);
	ac_max.z = fmax(ac_max.z, element.z);
	return ac_max;
}

float3 min_pt(float3 ac_min, float3 element) {
	ac_min.x = fmin(ac_min.x, element.x);
	ac_min.y = fmin(ac_min.y, element.y);
	ac_min.z = fmin(ac_min.z, element.z);
	return ac_min;
}

__kernel void min_max_reduction_start(__global PointXYZ* data, 
					__local float3* partial_min, 
					__local float3* partial_max,
					__global float3* data_min, 
					__global float3* data_max) {


	int lid = get_local_id(0);
	int group_size = get_local_size(0);
	partial_min[lid].x = data[get_global_id(0)].x;
	partial_min[lid].y = data[get_global_id(0)].y;
	partial_min[lid].z = data[get_global_id(0)].z; 	
	partial_max[lid] = partial_min[lid];

	barrier(CLK_LOCAL_MEM_FENCE);

	for (int i = group_size/2; i>0; i >>= 1) {
		if (lid < i) {
			partial_min[lid] = min_pt(partial_min[lid], partial_min[lid+i]);
			partial_max[lid] = max_pt(partial_max[lid], partial_max[lid+i]);
		}
		barrier(CLK_LOCAL_MEM_FENCE);
	}

	if (lid == 0) {
		data_min[get_group_id(0)] = partial_min[0];
		data_max[get_group_id(0)] = partial_max[0];
	}
}


__kernel void min_max_reduction(__global float3* data_min, 
				__global float3* data_max,
				__local float3* partial_min, 
				__local float3* partial_max) {

	int lid = get_local_id(0);
	int group_size = get_local_size(0);
	partial_min[lid] = data_min[get_global_id(0)];
 	partial_max[lid] = data_max[get_global_id(0)];

	barrier(CLK_LOCAL_MEM_FENCE);

	for (int i = group_size/2; i>0; i >>= 1) {
		if (lid < i) {
			partial_min[lid] = min_pt(partial_min[lid], partial_min[lid+i]);
			partial_max[lid] = max_pt(partial_max[lid], partial_max[lid+i]);
		}
		barrier(CLK_LOCAL_MEM_FENCE);
	}

	if (lid == 0) {
		data_min[get_group_id(0)] = partial_min[0];
		data_max[get_group_id(0)] = partial_max[0];
	}
}


__kernel void min_max_reduction_complete(__global float3* data_min, 
					 __global float3* data_max, 
					 __local float3* partial_min, 
					 __local float3* partial_max,
					 __global float3* output_min, 
					 __global float3* output_max) {
									
	int lid = get_local_id(0);
	int group_size = get_local_size(0);
	partial_min[lid] = data_min[get_global_id(0)];
 	partial_max[lid] = data_max[get_global_id(0)];

	barrier(CLK_LOCAL_MEM_FENCE);

	for (int i = group_size/2; i>0; i >>= 1) {
		if (lid < i) {
			partial_min[lid] = min_pt(partial_min[lid], partial_min[lid+i]);
			partial_max[lid] = max_pt(partial_max[lid], partial_max[lid+i]);
		}
		barrier(CLK_LOCAL_MEM_FENCE);
	}

	if (lid == 0) {
		output_min[get_group_id(0)] = partial_min[0];
		output_max[get_group_id(0)] = partial_max[0];
	}
}
