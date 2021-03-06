#ifdef MAC
#include <OpenCL/cl.hpp>
#else
#include <CL/cl.hpp>
#endif

typedef cl_float3 float3;
typedef cl_float4 float4;

namespace pcl
{
    namespace ocl
    {
        struct Morton
        {   
            const static int levels = 10;
            const static int bits_per_level = 3;
            const static int nbits = levels * bits_per_level;    

            typedef int code_t;

            static int spreadBits(int x, int offset)
            {
                //......................9876543210
                x = (x | (x << 10)) & 0x000f801f; //............98765..........43210
                x = (x | (x <<  4)) & 0x00e181c3; //........987....56......432....10
                x = (x | (x <<  2)) & 0x03248649; //......98..7..5..6....43..2..1..0
                x = (x | (x <<  2)) & 0x09249249; //....9..8..7..5..6..4..3..2..1..0

                return x << offset;
            }

            static int compactBits(int x, int offset)
            {                                      
                x = ( x >> offset ) & 0x09249249;  //....9..8..7..5..6..4..3..2..1..0
                x = (x | (x >>  2)) & 0x03248649;  //......98..7..5..6....43..2..1..0                                          
                x = (x | (x >>  2)) & 0x00e181c3;  //........987....56......432....10                                       
                x = (x | (x >>  4)) & 0x000f801f;  //............98765..........43210                                          
                x = (x | (x >> 10)) & 0x000003FF;  //......................9876543210        

                return x;
            }

            static code_t createCode(int cell_x, int cell_y, int cell_z)
            { 
                return spreadBits(cell_x, 0) | spreadBits(cell_y, 1) | spreadBits(cell_z, 2); 
            }

            static void decomposeCode(code_t code, int& cell_x, int& cell_y, int& cell_z)
            { 
                cell_x = compactBits(code, 0);
                cell_y = compactBits(code, 1);
                cell_z = compactBits(code, 2);        
            }

           static code_t extractLevelCode(code_t code, int level) 
           {
                return (code >> (nbits - 3 * (level + 1) )) & 7; 
           }

            static code_t shiftLevelCode(code_t level_code, int level)
            {
                return level_code << (nbits - 3 * (level + 1));
            }
        };

        struct CalcMorton
        {   
           const static int depth_mult = 1 << Morton::levels;

            float3 minp_;
            float3 dims_;    

            CalcMorton(float3 minp, float3 maxp) : minp_(minp) 
            {        
                dims_.x = maxp.x - minp.x;
                dims_.y = maxp.y - minp.y;
                dims_.z = maxp.z - minp.z;        
            }			

            Morton::code_t operator()(const float3& p) const
            {			
                int cellx = std::min((int)floor(depth_mult * (p.x - minp_.x)/dims_.x), depth_mult - 1);
                int celly = std::min((int)floor(depth_mult * (p.y - minp_.y)/dims_.y), depth_mult - 1);
                int cellz = std::min((int)floor(depth_mult * (p.z - minp_.z)/dims_.z), depth_mult - 1); 

                return Morton::createCode(cellx, celly, cellz);
            }
            /*	
            Morton::code_t operator()(const float4& p) const
            {			
                return (*this)(float3(p));                
            }
            */	
        };

        struct CompareByLevelCode
        {
            int level;

            CompareByLevelCode(int level_arg) : level(level_arg) {}    

            bool operator()(Morton::code_t code1, Morton::code_t code2) const 
            {                  
                return Morton::extractLevelCode(code1, level) < Morton::extractLevelCode(code2, level);  
            }	
        };
    }
}
