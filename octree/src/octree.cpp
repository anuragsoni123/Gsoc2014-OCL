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
 *  Author: Anatoly Baskeheev, Itseez Ltd, (myname.mysurname@mycompany.com)
 */

#include <pcl/ocl/octree/impl/octree.hpp>
#include <pcl/ocl/octree/impl/bitonic_sort.hpp>
#include <pcl/ocl/octree/impl/min_max.hpp>
#include <pcl/ocl/octree/impl/morton.hpp>
#include <pcl/ocl/octree/impl/morton_cl.hpp>
#include <pcl/ocl/octree/impl/octree_build.hpp>
#include<pcl/ocl/utils/ocl_manager.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include<assert.h>


using namespace std;
//////////////////////////////////////////////////////////////////////////////////////
//////////////// Octree Host Interface implementation ////////////////////////////////

pcl::ocl::Octree::Octree()
{
  last_level =1;
  num_nodes = 1;
  for(int i = 0; i < octree_level; i++)
    {
       last_level = last_level*8;
       num_nodes = num_nodes + last_level;
    }

  // 96 maximum leaves at last level which gives 5 levels ==> 37449
  begs = new int[num_nodes];   
  ends = new int[num_nodes];
  nodes = new int[num_nodes];
  parent = new int[num_nodes];
  codes = new int[num_nodes];
  
}

pcl::ocl::Octree::~Octree()
{
 clear();
}

void pcl::ocl::Octree::clear()
{
    delete begs;
    delete ends;
    delete nodes;
    delete parent;
    delete codes;
}

void pcl::ocl::Octree::setCloud(const PointCloud& cloud_arg)
{    
    cloud_ =  &cloud_arg;
}

void pcl::ocl::Octree::build()
{

  OCLManager* ocl = OCLManager::getInstance();
  cl_int err;

  cl::Event e_buf_cloud;
  cl::Buffer buf_cloud(ocl->getContext(), CL_MEM_READ_WRITE, sizeof(PointT) * cloud_->points.size(), NULL, &err);
  ocl->getQueue().enqueueWriteBuffer(buf_cloud, CL_TRUE, 0, sizeof(PointT) * cloud_->points.size(), const_cast<float*> (&cloud_->points[0].x), NULL, &e_buf_cloud);

	
//IMPLEMENT REDUCTION TO FIND MIN AND MAX VALUES OF THE CLOUD
  cl_float3 min, max;
  std::string PointFields = pcl::getFieldsList(*cloud_);
  get_min_max(buf_cloud, min, max, cloud_->points.size(),PointFields);
  octree_min = min;
  octree_max = max;
  //CALCULATE MORTON CODE and its LCA accross LEVELS OF EACH POINT
  morton_codes = new int[cloud_->points.size()];
  sort_key = new int[cloud_->points.size()];
  for(int i = 0; i < cloud_->points.size(); i++)
     sort_key[i] = i;

  calculate_morton(buf_cloud, cloud_->points.size(), min, max, morton_codes,PointFields);
  
  //SORT MORTON CODES
  bitonic_sort(morton_codes, sort_key, cloud_->points.size());
 
  // Building Octree
  octree_build(morton_codes, begs, ends, nodes, parent, codes, cloud_->points.size(), num_nodes, PointFields);

}

void pcl::ocl::Octree::approxNearestSearchHost(const PointType& query, int &out_index, float &sqr_dist)
{

    float3 minp = octree_min;
    float3 maxp = octree_max;
    float3 q;
    q.x = query.x;
    q.y = query.y;
    q.z = query.z;
    
    int node_idx = 0;

    bool out_of_root = query.x < minp.x || query.y < minp.y ||  query.z < minp.z || query.x > maxp.x || query.y > maxp.y ||  query.z > maxp.z;

    if(!out_of_root)        
    {
        pcl::ocl::CalcMorton morton_hpp(octree_min, octree_max);
  	int mor = morton_hpp(q);      

        pcl::ocl::Morton morton;
  
	int level_val, last;
        node_idx=num_nodes-last_level;
        last = last_level;
  	for(int i = 0; i < octree_level; i++)
   	{
     	last = last/8;
     	level_val = morton.extractLevelCode(mor, i);
     	node_idx += last * level_val;     
   	}

    }

    int beg = begs[node_idx];
    int end = ends[node_idx];

    sqr_dist = std::numeric_limits<float>::max();
    int out_index1;

    for(int i = beg; i <= end; ++i)
    {
        float dx = (cloud_->points[sort_key[i]].x - query.x);
        float dy = (cloud_->points[sort_key[i]].y - query.y);
        float dz = (cloud_->points[sort_key[i]].z - query.z);

        float d2 = dx * dx + dy * dy + dz * dz;

        if (sqr_dist > d2)
        {
            sqr_dist = d2;
            out_index1 = i;
        }
    }  

    out_index = sort_key[out_index1];



}

void pcl::ocl::Octree::calcBoundingBox(int level, int code, float3& res_minp, float3& res_maxp)
{        
    int cell_x, cell_y, cell_z;
    Morton::decomposeCode(code, cell_x, cell_y, cell_z);   

    float cell_size_x = (res_maxp.x - res_minp.x) / (1 << level);
    float cell_size_y = (res_maxp.y - res_minp.y) / (1 << level);
    float cell_size_z = (res_maxp.z - res_minp.z) / (1 << level);

    res_minp.x += cell_x * cell_size_x;
    res_minp.y += cell_y * cell_size_y;
    res_minp.z += cell_z * cell_size_z;

    res_maxp.x = res_minp.x + cell_size_x;
    res_maxp.y = res_minp.y + cell_size_y;
    res_maxp.z = res_minp.z + cell_size_z;       
}

bool pcl::ocl::Octree::checkIfNodeInsideSphere(const cl_float3& minp, const cl_float3& maxp, const cl_float3& c, float r)
{
    r *= r;

    float d2_xmin = (minp.x - c.x) * (minp.x - c.x);
    float d2_ymin = (minp.y - c.y) * (minp.y - c.y);
    float d2_zmin = (minp.z - c.z) * (minp.z - c.z);

    if (d2_xmin + d2_ymin + d2_zmin > r)
        return false;

    float d2_zmax = (maxp.z - c.z) * (maxp.z - c.z);

    if (d2_xmin + d2_ymin + d2_zmax > r)
        return false;

    float d2_ymax = (maxp.y - c.y) * (maxp.y - c.y);

    if (d2_xmin + d2_ymax + d2_zmin > r)
        return false;

    if (d2_xmin + d2_ymax + d2_zmax > r)
        return false;

    float d2_xmax = (maxp.x - c.x) * (maxp.x - c.x);

    if (d2_xmax + d2_ymin + d2_zmin > r)
        return false;

    if (d2_xmax + d2_ymin + d2_zmax > r)
        return false;

    if (d2_xmax + d2_ymax + d2_zmin > r)
        return false;

    if (d2_xmax + d2_ymax + d2_zmax > r)
        return false;

    return true;
}


int pcl::ocl::Octree::getBitsNum(int interger)
{
   int count = 0;
   while(interger > 0)
   {
    if (interger & 1)
	++count;
    interger>>=1;
   }
   return count;
} 

bool pcl::ocl::Octree::checkIfNodeOutsideSphere(const cl_float3& minp, const cl_float3& maxp, const cl_float3& c, float r)
{
    if (maxp.x < (c.x - r) ||  maxp.y < (c.y - r) || maxp.z < (c.z - r))
        return true;

    if ((c.x + r) < minp.x || (c.y + r) < minp.y || (c.z + r) < minp.z)
        return true;

    return false;
}


void pcl::ocl::Octree::radiusSearchHost(const PointType& query, float radius, vector<int>& out, int max_nn)
{

  cl_float3 center;
  center.x = query.x;
  center.y = query.y;
  center.z = query.z;
  
  pcl::ocl::OctreeIteratorHost iterator;
  while(iterator.level >= 0)
    {        
        int node_idx = *iterator;
        int code = codes[node_idx];

        float3 node_minp = octree_min;
        float3 node_maxp = octree_max;        
        calcBoundingBox(iterator.level, code, node_minp, node_maxp);

        if (checkIfNodeOutsideSphere(node_minp, node_maxp, center, radius))        
        {                
            ++iterator;            
            continue;
        }

        //if true, take all, and go to next
        if (checkIfNodeInsideSphere(node_minp, node_maxp, center, radius))
        {            
            int beg = begs[node_idx];
            int end = ends[node_idx];
            int diff = end-beg+(int)out.size();
            if(diff > max_nn)
               diff = max_nn;
            end = beg + diff - (int)out.size();

            out.insert(out.end(), sort_key[beg], sort_key[end]);
            if (out.size() == (size_t)max_nn)
                return;

            ++iterator;
            continue;
        }

        // test children
        int children_mask = nodes[node_idx] & 0xFF;

        bool isLeaf = children_mask == 0;

        if (isLeaf)
        {            
            const int beg = begs[node_idx];
            const int end = ends[node_idx];                                    

            for(int j = beg; j < end; ++j)
            {
                int index = sort_key[j];//host_octree.indices[j];
                float point_x = cloud_->points[index].x;
                float point_y = cloud_->points[index].y;
                float point_z = cloud_->points[index].z;

                float dx = (point_x - center.x);
                float dy = (point_y - center.y);
                float dz = (point_z - center.z);

                float dist2 = dx * dx + dy * dy + dz * dz;

                if (dist2 < radius * radius)
                    out.push_back(index);

                if (out.size() == (size_t)max_nn)
                    return;
            }               
            ++iterator;               
            continue;
        }

        int first  = nodes[node_idx] >> 8;        
        iterator.gotoNextLevel(first, getBitsNum(children_mask));                
    } 
  
  
  
}

