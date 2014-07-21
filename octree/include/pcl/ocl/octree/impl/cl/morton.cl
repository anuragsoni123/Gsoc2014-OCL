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

typedef int code_t;

typedef struct
	{   
            int levels;
            int bits_per_level;
            int nbits;
	}Morton;

                int spreadBits(int x, int offset)
            {
                //......................9876543210
                x = (x | (x << 10)) & 0x000f801f; //............98765..........43210
                x = (x | (x <<  4)) & 0x00e181c3; //........987....56......432....10
                x = (x | (x <<  2)) & 0x03248649; //......98..7..5..6....43..2..1..0
                x = (x | (x <<  2)) & 0x09249249; //....9..8..7..5..6..4..3..2..1..0

                return x << offset;
            }

                int compactBits(int x, int offset)
            {                                      
                x = ( x >> offset ) & 0x09249249;  //....9..8..7..5..6..4..3..2..1..0
                x = (x | (x >>  2)) & 0x03248649;  //......98..7..5..6....43..2..1..0                                          
                x = (x | (x >>  2)) & 0x00e181c3;  //........987....56......432....10                                       
                x = (x | (x >>  4)) & 0x000f801f;  //............98765..........43210                                          
                x = (x | (x >> 10)) & 0x000003FF;  //......................9876543210        

                return x;
            }

                code_t createCode(int cell_x, int cell_y, int cell_z)
            { 
                return spreadBits(cell_x, 0) | spreadBits(cell_y, 1) | spreadBits(cell_z, 2); 
            }

                void decomposeCode(code_t code, int cell_x, int cell_y, int cell_z)
            { 
                cell_x = compactBits(code, 0);
                cell_y = compactBits(code, 1);
                cell_z = compactBits(code, 2);        
            }

            code_t extractLevelCode(Morton morton, code_t code, int level) 
            {
                return (code >> (morton.nbits - 3 * (level + 1) )) & 7; 
            }

            code_t shiftLevelCode(Morton morton, code_t level_code, int level)
            {
                return level_code << (morton.nbits - 3 * (level + 1));
            }

        typedef struct
        {   
	    int depth_mult;
            float3 minp_;
            float3 dims_;    
} CalcMorton;

            CalcMorton dimsMorton(float3 minp, float3 maxp)
            {
		CalcMorton calcmorton;
		calcmorton.minp_ = minp;        
                calcmorton.dims_.x = maxp.x - minp.x;
                calcmorton.dims_.y = maxp.y - minp.y;
                calcmorton.dims_.z = maxp.z - minp.z;
		return calcmorton;        
            }			

            code_t get_code(CalcMorton calcmorton, PointXYZ p)
            {			
                int cellx = min((int)floor(calcmorton.depth_mult * (p.x - calcmorton.minp_.x)/calcmorton.dims_.x), calcmorton.depth_mult - 1);
                int celly = min((int)floor(calcmorton.depth_mult * (p.y - calcmorton.minp_.y)/calcmorton.dims_.y), calcmorton.depth_mult - 1);
                int cellz = min((int)floor(calcmorton.depth_mult * (p.z - calcmorton.minp_.z)/calcmorton.dims_.z), calcmorton.depth_mult - 1); 

                return createCode(cellx, celly, cellz);
            }

int get_lca_code(Morton morton, int morton_cur, int morton_nxt)
	{
		int lca = 0;
		int level =0;
		int p_lca_cur, p_lca_nxt;
		bool equals = true;
		while((level<=morton.levels) && equals)
		{
			p_lca_cur = extractLevelCode(morton, morton_cur, level);
			p_lca_nxt = extractLevelCode(morton, morton_nxt, level);
			equals = (p_lca_cur == p_lca_nxt) ? true : false;
			level++;
		}
                level--;
		return morton_cur>>((morton.levels-level)*morton.bits_per_level);
	}
__kernel void calculate_morton(__global PointXYZ* data,
				float3 min,
				float3 max, 
			       __global int* code_array) 
{
	Morton morton;
	morton.levels = 10;
        morton.bits_per_level = 3;
        morton.nbits = morton.levels * morton.bits_per_level;
	
	CalcMorton calcmorton = dimsMorton(min, max);
	calcmorton.depth_mult = 1 << morton.levels;
	code_array[get_global_id(0)] = get_code(calcmorton, data[get_global_id(0)]);
}

__kernel void calculate_lca(__global int* morton_codes, 
			    __global int* lca_codes) 
{
	Morton morton;
	morton.levels = 10;
        morton.bits_per_level = 3;
        morton.nbits = morton.levels * morton.bits_per_level;
	lca_codes[get_global_id(0)] = get_lca_code(morton, morton_codes[get_global_id(0)], morton_codes[get_global_id(0)+1]);
}

int divUp(int total, int grain)
{ 
	return (total + grain - 1) / grain;
}


int comp(code_t first, code_t second, Morton morton, int level)
{
  int first_code  = extractLevelCode(morton, first, level);
  int second_code = extractLevelCode(morton, second, level);
  if(first_code < second_code)
    return 1;
  
return 0; 
}

int lower_bound( __global int * codes, int first, int last, int val, Morton morton, int level)
{  
    int len = last - first;
    int val_comp;
    while(len > 0)
    {
        int half_length = len >> 1;
        int middle = first;

        middle += half_length;
        val_comp = codes[middle];
        if(comp(val_comp, val, morton, level))
        {
            first = middle;
            ++first;
            len = len - half_length - 1;
        }
        else
        {
            len = half_length;
        }
    }
    return first;
}

int FindCells(Morton morton, __global int* codes, __global int* octree_begs, __global int * octree_ends, int task, int level, int *cell_begs, char *cell_code) 
{               
	int cell_count = 0;
	int beg = octree_begs[task];
	int end = octree_ends[task];                
        int max_points_per_leaf = 96;
                                 
	if (end - beg < max_points_per_leaf)
	{   
	    //cell_count == 0;
	}
	else
	{
            int cur_code = codes[beg];
            int pos;
	    cur_code = extractLevelCode(morton,cur_code,level);
	    cell_begs[cell_count] = beg;
	    cell_code[cell_count] = cur_code;     
	    ++cell_count;                        
            int last_code = codes[end - 1];
            last_code = extractLevelCode(morton,codes[end - 1], level);
	    if (last_code == cur_code)
	    {
		cell_begs[cell_count] = end;                                         
	    }
	    else
	    {
		for(;;)
		{
		    int search_code = cur_code + 1;
		    if (search_code == 8)
		    {
		        cell_begs[cell_count] = end+1;
		        break;
		    }

		    int morton_code = shiftLevelCode(morton, search_code, level);
		    pos = lower_bound(codes, beg, end, morton_code, morton, level);
                    /*if(search_code == 7 && task == 1 && level == 1)
		    {
                    octree_begs[0+9] = cur_code;
		    octree_begs[1+9] = beg;
                    octree_begs[2+9] = codes[beg];
		    octree_begs[3+9] = search_code;
                    octree_begs[4+9] = pos;
                    octree_begs[5+9] = level;
                    octree_begs[6+9] = end;
		    //octree_begs[7+9] = codes[pos];
		    //octree_begs[8+9] = codes[pos+1];
		    }*/		                    
		    
                    if (pos == end)
		    {
		        cell_begs[cell_count] = end;
		        break;
		    }
          
                    cur_code = codes[pos];
		    cur_code = extractLevelCode(morton, cur_code, level);
                    cell_begs[cell_count] = pos;
		    cell_code[cell_count] = cur_code;
		    ++cell_count;
		    beg = pos;  
		                      
		}       
	    }
	}
	return cell_count;
}

__kernel void octree_build(__global int* morton_codes, __global int* octree_begs, __global int* octree_ends,__global int* octree_nodes, __global int* octree_parent, __global int * octree_codes) 
{

	Morton morton;
	morton.levels = 10;
        morton.bits_per_level = 3;
        morton.nbits = morton.levels * morton.bits_per_level;
        const int max_points_per_leaf = 96;
        __local int nodes_num;
        __local int tasks_beg;
        __local int tasks_end;
        __local int total_new;
        __local int offsets[1024];
        int level = 0;
        int CTA_SIZE = 1024;
        int threadIdx = get_global_id(0);
        
	if (threadIdx == 0)
        {
            //init root
            octree_codes[0] = 0;
            octree_nodes[0] = 0;
            octree_begs[0] = 0;
            octree_ends[0] = 1024*1024-1;
            octree_parent[0] = -1;

            //init shared                    
            nodes_num = 1;
            tasks_beg = 0;
            tasks_end = 1;
            total_new = 0;
        }
      
        
        barrier(CLK_LOCAL_MEM_FENCE);  
        while (tasks_beg < tasks_end)
        {
	   
	   int  cell_begs[9];   // 8 + 1 ==> 8 is number of children
           char cell_code[8];
	   int task_count = tasks_end - tasks_beg;                    
           int iters = divUp(task_count, CTA_SIZE);

           int task = tasks_beg + threadIdx;
	   for(int it = 0; it < iters; ++it, task += CTA_SIZE)
            {		
		int cell_count = 0;
                if(task < tasks_end )
                  cell_count = FindCells(morton, morton_codes, octree_begs, octree_ends, task, level, cell_begs, cell_code);

                offsets[threadIdx] = cell_count;  
		barrier(CLK_LOCAL_MEM_FENCE);
                
		if(threadIdx == 0)
		{ 
		 int sum = 0;
 		 int prev;
                 for(int i = 0; i < 1024; i++)
		  {
                   prev = offsets[i]; 
		   offsets[i] = sum;
                   sum = sum + prev;
		  }                
		}  
		barrier(CLK_LOCAL_MEM_FENCE);
              
		if (task < tasks_end)
                {
		    if (cell_count > 0)
                    {
                        int parent_code_shifted = octree_codes[task] << 3;
                        int offset = nodes_num + offsets[threadIdx];
			
                        int mask = 0;
                        for(int i = 0; i < cell_count; ++i)
                        {
                            octree_begs [offset + i] = cell_begs[i];
                            octree_ends [offset + i] = cell_begs[i + 1]-1;
                            octree_codes[offset + i] = parent_code_shifted + cell_code[i];

                            octree_parent[offset + i] = task;
                            mask |= (1 << cell_code[i]);
                        }
                        octree_nodes[task] = (offset << 8) + mask;
                    }
                    else
                        octree_nodes[task] = 0;
                }
		barrier(CLK_LOCAL_MEM_FENCE);
		if (threadIdx == CTA_SIZE - 1)
                {                            
                    total_new += cell_count + offsets[threadIdx];
                    nodes_num += cell_count + offsets[threadIdx];
                }
		offsets[threadIdx] = 0;  
		barrier(CLK_LOCAL_MEM_FENCE);    
                
	     }

             if (threadIdx == CTA_SIZE - 1)
             {                       
                tasks_beg  = tasks_end;
                tasks_end += total_new;        
                total_new = 0;
             }
                ++level;
                
	   barrier(CLK_LOCAL_MEM_FENCE);
                
	}        
           //if (threadIdx.x == CTA_SIZE - 1)
               //*octree_nodes_num = nodes_num;

}
