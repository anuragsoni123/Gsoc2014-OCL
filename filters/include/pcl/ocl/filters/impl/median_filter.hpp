/*
 * Software License Agreement (BSD License)
 *
 * Point Cloud Library (PCL) - www.pointclouds.org
 * Copyright (c) 2012-, Open Perception, Inc.

 * All rights reserved.

 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above
 *    copyright notice, this list of conditions and the following
 *    disclaimer in the documentation and/or other materials provided
 *    with the distribution.
 * * Neither the name of the copyright holder(s) nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 * FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 * COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 * INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 * ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 *
 * $Id$
 *
 */

#ifndef PCL_OCL_FILTERS_IMPL_MEDIAN_FILTER_HPP_
#define PCL_OCL_FILTERS_IMPL_MEDIAN_FILTER_HPP_

#include <pcl/ocl/filters/median_filter.h>
#include <pcl/common/io.h>
#include <stdio.h>
#include<pcl/ocl/utils/ocl_manager.h>
#include<CL/cl.hpp>
#include <iostream>
#include <fstream>
#include <string>
#include<math.h>
using namespace cl;

template <typename PointT> void
pcl::ocl::MedianFilter<PointT>::applyFilter (PointCloud &output)
{
  if (!input_->isOrganized ())
  {
    PCL_ERROR ("[pcl::MedianFilter] Input cloud needs to be organized\n");
    return;
  }

  // Copy everything from the input cloud to the output cloud (takes care of all the fields)
  copyPointCloud (*input_, output);
 
  int height = static_cast<int> (output.height);
  int width = static_cast<int> (output.width);
  
  cl_int ret;
  OCLManager *test1;
  test1 = test1->getInstance();

  Context context = test1->getContext();
  CommandQueue queue = test1->getQueue();
  
  std::string sourceFile("median_filter.clbin");
  Program program = test1->buildProgramFromBinary(sourceFile);  
  
  // Make kernel
  Kernel kernel(program, "median_filter_kernel",&ret);
    
  // Create memory buffers
  Buffer bufferA = Buffer(context, CL_MEM_READ_ONLY,  height * width * sizeof(PointT));
  Buffer bufferB = Buffer(context, CL_MEM_WRITE_ONLY, height * width * sizeof(PointT));

  ret = queue.enqueueWriteBuffer(bufferA, CL_TRUE, 0, height * width* sizeof(PointT),&(input_->points[0].x));
  ret = queue.enqueueWriteBuffer(bufferB, CL_TRUE, 0, height * width * sizeof(PointT), &(output.points[0]));
  
  // Set arguments to kernel
  ret = kernel.setArg(0, bufferA);
  ret = kernel.setArg(1, bufferB);
  
  ret = kernel.setArg(2, sizeof(int), (void *)&width);
  ret = kernel.setArg(3, sizeof(int), (void *)&height);
  ret = kernel.setArg(4, sizeof(int), (void *)&window_size_);
  ret = kernel.setArg(5, sizeof(float), (void *)&max_allowed_movement_);


  NDRange global(height,width);
  NDRange local(1,1);
  ret = queue.enqueueNDRangeKernel(kernel,NullRange, global,local);
  
  ret = queue.enqueueReadBuffer(bufferB, CL_TRUE, 0, height * width * sizeof(PointT)  , &(output.points[0]));
  test1->destroyInstance();
    
}


#endif /* PCL_FILTERS_IMPL_MEDIAN_FILTER_HPP_ */
