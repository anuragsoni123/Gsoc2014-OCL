/*
 * Software License Agreement (BSD License)
 *
 *  Point Cloud Library (PCL) - www.pointclouds.org
 *  Copyright (c) 2010-2012, Willow Garage, Inc.
 *
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
 *   * Neither the name of the copyright holder(s) nor the names of its
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
 * $Id$
 *
 */

#ifndef PCL_OCL_FILTERS_IMPL_PASSTHROUGH_HPP_
#define PCL_OCL_FILTERS_IMPL_PASSTHROUGH_HPP_

#include <pcl/ocl/filters/passthrough.h>
#include <pcl/common/io.h>
#include<pcl/ocl/utils/ocl_manager.h>
using namespace cl;
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename PointT> void
pcl::ocl::PassThrough<PointT>::applyFilter (PointCloud &output)
{
  std::vector<int> indices;
  removed_indices_->resize (indices_->size ());
  if (keep_organized_)
  { 
    bool temp = extract_removed_indices_;
    extract_removed_indices_ = true;
    applyFilterIndices (indices);
    extract_removed_indices_ = temp;
    output = *input_;
    for (int rii = 0; rii < static_cast<int> (removed_indices_->size ()); ++rii)  // rii = removed indices iterator
      output.points[(*removed_indices_)[rii]].x = output.points[(*removed_indices_)[rii]].y = output.points[(*removed_indices_)[rii]].z = user_filter_value_;
    if (!pcl_isfinite (user_filter_value_))
      output.is_dense = false;
  }
  else
  {
    output.is_dense = true;
    applyFilterIndices (indices);
    int oii=0, rii=0;
    output.points.resize (indices.size());
    output.header   = input_->header;
    output.height   = 1;
    output.is_dense = input_->is_dense;
    output.sensor_orientation_ = input_->sensor_orientation_;
    output.sensor_origin_ = input_->sensor_origin_;
   
    for(int i =0; i<10;i++)
      cout << indices[i];
    for(int i =0; i<indices.size(); i++)
     { 
        if(indices[i]==1) 
          output.points[oii++] = input_->points[i];
        else
        {
          if (extract_removed_indices_)
            (*removed_indices_)[rii++]=i;
        }
          
     }
       output.points.resize(oii);
       output.width = oii;
       removed_indices_->resize (rii);     
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename PointT> void
pcl::ocl::PassThrough<PointT>::applyFilterIndices (std::vector<int> &indices)
{
  // The arrays to be used
  indices.resize (indices_->size ());
  removed_indices_->resize (indices_->size ());
  
  OCLManager *pt;
  pt = pt->getInstance();

  Context context = pt->getContext();
  CommandQueue queue = pt->getQueue();

  std::string sourceFile("passthrough.clbin");
  Program program = pt->buildProgramFromBinary(sourceFile);  
  
  
  // Has a field name been specified?
  if (filter_field_name_.empty ())
  {
     // Make kernel
    Kernel kernel(program, "remove_NAN_kernel");
    int size = indices_->size();
    cout << "\n\ninside Remove NAN";
   // Create memory buffers
    Buffer bufferInput(context, CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR,  input_->height * input_->width * sizeof(PointT), (void *)&(input_->points[0]));
    
    Buffer buffer_indices(context, CL_MEM_WRITE_ONLY|CL_MEM_COPY_HOST_PTR, size * sizeof(int), &indices[0]);  
    
    //Buffer points(context, CL_MEM_READ_WRITE|CL_MEM_COPY_HOST_PTR, 2*sizeof(int), total_points);  
    
    // Set arguments to kernel
    kernel.setArg(0, bufferInput);
    kernel.setArg(1, buffer_indices);
    kernel.setArg(2, sizeof(int), (void *)&size);
    int BlockSize = 1024;
    
    int SizeX = ((size+BlockSize-1)/BlockSize)*BlockSize;
   
    NDRange global(SizeX);
    NDRange local(BlockSize);
   
    queue.enqueueNDRangeKernel(kernel,NullRange, global,local);
    queue.enqueueReadBuffer(buffer_indices, CL_TRUE, 0, size * sizeof(int), &(indices[0]));
  }
  else
  {
    // Attempt to get the field name's index
    std::vector<pcl::PCLPointField> fields;
    int distance_idx = pcl::getFieldIndex (*input_, filter_field_name_, fields);

    //cout << "\n " << distance_idx;
    if (distance_idx == -1)
    {
      PCL_WARN ("[pcl::%s::applyFilter] Unable to find field name in point type.\n", getClassName ().c_str ());
      indices.clear ();
      removed_indices_->clear ();
      return;
    }

    Kernel kernel(program, "passthrough");
    int size = indices_->size();
     // Create memory buffers
    Buffer bufferInput(context, CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR,  input_->height * input_->width * sizeof(PointT), (void *)&(input_->points[0]));
    Buffer buffer_indices(context, CL_MEM_WRITE_ONLY|CL_MEM_COPY_HOST_PTR, size * sizeof(int), &indices[0]);  
 
    // Set arguments to kernel
    kernel.setArg(0, bufferInput);
    kernel.setArg(1, buffer_indices);
    kernel.setArg(2, sizeof(int), (void *)&size);
    kernel.setArg(3, sizeof(int), (void *)&distance_idx);
    kernel.setArg(4, sizeof(float), (void *)&filter_limit_min_);
    kernel.setArg(5, sizeof(float), (void *)&filter_limit_max_);
    kernel.setArg(6, sizeof(int), (void *)&negative_);
    
    int BlockSize = 1024;
    int SizeX = ((size+BlockSize-1)/BlockSize)*BlockSize;
   
    NDRange global(SizeX);
    NDRange local(BlockSize);
    cl_uint ret;
    queue.enqueueNDRangeKernel(kernel,NullRange, global,local);
    //queue.enqueueReadBuffer(points, CL_TRUE, 0,  2*sizeof(int), total_points);
    queue.enqueueReadBuffer(buffer_indices, CL_TRUE, 0, size * sizeof(int), &(indices[0]));
    
  }

  pt->destroyInstance();  
}

#define PCL_INSTANTIATE_PassThroughOCL(T) template class PCL_EXPORTS pcl::ocl::PassThrough<T>;

#endif  // PCL_FILTERS_IMPL_PASSTHROUGH_HPP_

