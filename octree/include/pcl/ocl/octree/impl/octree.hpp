/*
 * Software License Agreement (BSD License)
 *
 *  Copyright (c) 2011, Willow Garage, Inc.
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   * Neither the name of Willow Garage, Inc. nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *
 */

#ifndef _PCL_OCL_OCTREE_
#define _PCL_OCL_OCTREE_

#ifdef MAC
#include <OpenCL/cl.hpp>
#else
#include <CL/cl.hpp>
#endif

#include <pcl/point_types.h>
#include <pcl/pcl_macros.h>
#include <pcl/point_cloud.h>

namespace pcl
{
    namespace ocl
    {   
        /**
         * \brief   Octree implementation using OpenCL. It suppors parallel building and parallel search as well .       
         */

        class Octree
        {
        public:

            /** \brief Default constructor.*/             
            Octree();
           
            /** \brief Denstructor.*/             
            virtual ~Octree();

            /** \brief Point typwe supported */
            typedef pcl::PointXYZ PointType;

            typedef pcl::PointCloud<PointType> PointCloud;
            
            const PointCloud* cloud_;
            
            /** \brief Sets cloud for which octree is built */            
            void setCloud(const PointCloud& cloud_arg);

            /** \brief Performs parallel octree building */
	    void build();

            /** \brief Performs approximate neares neighbor search on CPU. It call \a internalDownload if nessesary
              * \param[in]  query 3D point for which neighbour is be fetched             
              * \param[out] out_index neighbour index
              * \param[out] sqr_dist square distance to the neighbour returned
              */

            int *nodes;
            int *codes;
            int *begs;
            int *ends;

            int *nodes_num;

            int *parent;
            
            /** \brief Performs search of all points wihtin given radius on CPU. It call \a internalDownload if nessesary
              * \param[in] center center of sphere
              * \param[in] radius radious of sphere
              * \param[out] out indeces of points within give sphere
              * \param[in] max_nn maximum numver of results returned
              */
            void radiusSearchHost(const PointType& center, float radius, std::vector<int>& out, int max_nn = INT_MAX);

            /** \brief Performs approximate neares neighbor search on CPU. It call \a internalDownload if nessesary
              * \param[in]  query 3D point for which neighbour is be fetched             
              * \param[out] out_index neighbour index
              * \param[out] sqr_dist square distance to the neighbour returned
              */
            void approxNearestSearchHost(const PointType& query, int &out_index, float &sqr_dist);
	     //void approxNearestSearchHost(const PointType& query, std::vector<int> &out_index, std::vector<float> &sqr_dist);

            /** \brief Performs batch radius search on GPU
              * \param[in] centers array of centers 
              * \param[in] radius radius for all queries
              * \param[in] max_results max number of returned points for each querey
              * \param[out] result results packed to signle array
              */
            //void radiusSearch(const Queries& centers, float radius, int max_results, NeighborIndices& result) const;

            /** \brief Performs batch radius search on GPU
              * \param[in] centers array of centers 
              * \param[in] radiuses array of radiuses
              * \param[in] max_results max number of returned points for each querey
              * \param[out] result results packed to signle array
              */
            //void radiusSearch(const Queries& centers, const Radiuses& radiuses, int max_results, NeighborIndices& result) const;

            /** \brief Performs batch radius search on GPU
              * \param[in] centers array of centers  
              * \param[in] indices indices for centers array (only for these points search is performed)
              * \param[in] radius radius for all queries
              * \param[in] max_results max number of returned points for each querey
              * \param[out] result results packed to signle array
              */
            //void radiusSearch(const Queries& centers, const Indices& indices, float radius, int max_results, NeighborIndices& result) const;

            /** \brief Batch approximate nearest search on GPU
              * \param[in] queries array of centers
              * \param[out] result array of results ( one index for each query ) 
              */
            //void approxNearestSearch(const Queries& queries, NeighborIndices& result) const;

            /** \brief Batch exact k-nearest search on GPU for k == 1 only!
              * \param[in] queries array of centers
              * \param[in] k nubmer of neighbors (only k == 1 is supported)
              * \param[out] results array of results
              */
            //void nearestKSearchBatch(const Queries& queries, int k, NeighborIndices& results) const;

            /** \brief Desroys octree and release all resources */
            void clear();            

        private:
	    const static int octree_level=5;
            void calcBoundingBox(int level, int code, cl_float3& res_minp, cl_float3& res_maxp);
            bool checkIfNodeOutsideSphere(const cl_float3& minp, const cl_float3& maxp, const cl_float3& c, float r);
	    bool checkIfNodeInsideSphere(const cl_float3& minp, const cl_float3& maxp, const cl_float3& c, float r);
	    cl_float3 octree_min, octree_max;
            int *morton_codes, *sort_key;
            int num_nodes, last_level;
            int getBitsNum(int integer);
            	    
        };

	struct OctreeIteratorHost
        {        
        const static int MAX_LEVELS_PLUS_ROOT = 11;
        int paths[MAX_LEVELS_PLUS_ROOT];          
        int level;

        OctreeIteratorHost()
        {
            level = 0; // root level
            paths[level] = (0 << 8) + 1;                    
        }

        void gotoNextLevel(int first, int len) 
        {   
            ++level;
            paths[level] = (first << 8) + len;        
        }       

        int operator*() const 
        { 
            return paths[level] >> 8; 
        }        

        void operator++()
        {
            while(level >= 0)
            {
                int data = paths[level];

                if ((data & 0xFF) > 1) // there are another siblings, can goto there
                {                           
                    data += (1 << 8) - 1;  // +1 to first and -1 from len
                    paths[level] = data;
                    break;
                }
                else
                    --level; //goto parent;            
            }        
        }        
      };        

    }
}

#endif /* _PCL_GPU_OCTREE_ */
