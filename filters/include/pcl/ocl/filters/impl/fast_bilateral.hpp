/*
 * Software License Agreement (BSD License)
 *
 * Point Cloud Library (PCL) - www.pointclouds.org
 * Copyright (c) 2012-, Open Perception, Inc.
 * Copyright (c) 2004, Sylvain Paris and Francois Sillion

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
#ifndef PCL_OCL_FILTERS_IMPL_FAST_BILATERAL_HPP_
#define PCL_OCL_FILTERS_IMPL_FAST_BILATERAL_HPP_

#include <pcl/common/io.h>
#include<pcl/ocl/utils/ocl_manager.h>
#include <pcl/ocl/filters/fast_bilateral.h>
using namespace cl;
//////////////////////////////////////////////////////////////////////////////////////////////
template <typename PointT> void
pcl::ocl::FastBilateralFilter<PointT>::applyFilter (PointCloud &output)
{
  if (!input_->isOrganized ())
  {
    PCL_ERROR ("[pcl::FastBilateralFilter] Input cloud needs to be organized.\n");
    return;
  }

  std::size_t x,y;
  OCLManager *bf;
  bf = bf->getInstance();

  Context context = bf->getContext();
  CommandQueue queue = bf->getQueue();
  std::string sourceFile("fast_bilateral.clbin");
  Program program = bf->buildProgramFromBinary(sourceFile);  
  
  copyPointCloud (*input_, output);
  float base_max = -std::numeric_limits<float>::max (),
        base_min = std::numeric_limits<float>::max ();
if(input_->is_dense)
{
  int size = input_->height * input_->width;
  int BlockSize = 1024;
  // Make kernel
  Kernel kernel(program, "reduction_cloud");

  // Create memory buffers
  Buffer bufferA(context, CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR,  size * sizeof(PointT), (void *)&(input_->points[0]));
  
  LocalSpaceArg l_max = cl::__local(BlockSize * sizeof(float));
  LocalSpaceArg l_min = cl::__local(BlockSize * sizeof(float));
  
  int SizeX = (size/BlockSize)*BlockSize;
  int outSize = SizeX/BlockSize;
  Buffer buffermax(context, CL_MEM_READ_WRITE, outSize * sizeof(float));
  Buffer buffermin(context, CL_MEM_READ_WRITE, outSize * sizeof(float));
 
  
  // Set arguments to kernel
  kernel.setArg(0, bufferA);
  kernel.setArg(1, l_max);
  kernel.setArg(2, l_min);
  kernel.setArg(3, buffermax);
  kernel.setArg(4, buffermin);
 
  NDRange global(SizeX);
  NDRange local(BlockSize);
  
  queue.enqueueNDRangeKernel(kernel,NullRange, global,local);

  float maxi[outSize];
  float mini[outSize];
  
  queue.enqueueReadBuffer(buffermax, CL_TRUE, 0, outSize * sizeof(float), &(maxi[0]));
  queue.enqueueReadBuffer(buffermin, CL_TRUE, 0, outSize * sizeof(float), &(mini[0]));
  
 base_max = maxi[0];
 base_min = mini[0];
 for (int i = 1; i < outSize ; i ++)
 {  
    if (base_max < maxi[i])
       base_max = maxi[i];
    if (base_min > mini[i])
        base_min = mini[i]; 
    
  }

}
else
{  
  
  int size = input_->height * input_->width;
  int BlockSize = 1024;
  // Make kernel
  Kernel kernel(program, "reduction_nan_cloud");

  // Create memory buffers
  Buffer bufferA(context, CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR,  size * sizeof(PointT), (void *)&(input_->points[0]));
  
  LocalSpaceArg l_max = cl::__local(BlockSize * sizeof(float));
  LocalSpaceArg l_min = cl::__local(BlockSize * sizeof(float));
  
  int SizeX = (size/BlockSize)*BlockSize;
  int outSize = SizeX/BlockSize;
  Buffer buffermax(context, CL_MEM_READ_WRITE, outSize * sizeof(float));
  Buffer buffermin(context, CL_MEM_READ_WRITE, outSize * sizeof(float));
 
  
  // Set arguments to kernel
  kernel.setArg(0, bufferA);
  kernel.setArg(1, l_max);
  kernel.setArg(2, l_min);
  kernel.setArg(3, buffermax);
  kernel.setArg(4, buffermin);
 
  NDRange global(SizeX);
  NDRange local(BlockSize);
  
  queue.enqueueNDRangeKernel(kernel,NullRange, global,local);

  float maxi[outSize];
  float mini[outSize];
  
  queue.enqueueReadBuffer(buffermax, CL_TRUE, 0, outSize * sizeof(float), &(maxi[0]));
  queue.enqueueReadBuffer(buffermin, CL_TRUE, 0, outSize * sizeof(float), &(mini[0]));
  
 //base_max = maxi[0];
 //base_min = mini[0];
 for (int i = 0; i < outSize ; i ++)
 {  
    if(isfinite(maxi[i]))
	{
    if (base_max < maxi[i])
       base_max = maxi[i];
    if (base_min > mini[i])
        base_min = mini[i]; 
	}  
}

 for (int i = 0; i < input_->height * input_->width; ++i)
   {
      if(!pcl_isfinite(output[i].z))
      output[i].z = base_max;
   }
 
}
  const float base_delta = base_max - base_min;

  const std::size_t padding_xy = 2;
  const std::size_t padding_z  = 2;

  const std::size_t small_width  = static_cast<std::size_t> (static_cast<float> (input_->width  - 1) / sigma_s_) + 1 + 2 * padding_xy;
  const std::size_t small_height = static_cast<std::size_t> (static_cast<float> (input_->height - 1) / sigma_s_) + 1 + 2 * padding_xy;
  const std::size_t small_depth  = static_cast<std::size_t> (base_delta / sigma_r_)   + 1 + 2 * padding_z;


  Array3D data (small_width, small_height, small_depth);
  for (y = 0; y < input_->height; ++y)
  {
    //const size_t small_x = static_cast<size_t> (static_cast<float> (x) / sigma_s_ + 0.5f) + padding_xy;
    const std::size_t small_y = static_cast<std::size_t> (static_cast<float> (y) / sigma_s_ + 0.5f) + padding_xy;
    for ( x = 0; x < input_->width; ++x)
    {
      const std::size_t small_x = static_cast<std::size_t> (static_cast<float> (x) / sigma_s_ + 0.5f) + padding_xy;
      const float z = output (x,y).z - base_min;
      const std::size_t small_z = static_cast<std::size_t> (static_cast<float> (z) / sigma_r_ + 0.5f) + padding_z;

      Eigen::Vector2f& d = data (small_x, small_y, small_z);
      d[0] += output (x,y).z;
      d[1] += 1.0f;
    }
  }


  std::vector<long int> offset (3);
  offset[0] = &(data (1,0,0)) - &(data (0,0,0));
  offset[1] = &(data (0,1,0)) - &(data (0,0,0));
  offset[2] = &(data (0,0,1)) - &(data (0,0,0));

  Array3D buffer (small_width, small_height, small_depth);

  for (std::size_t dim = 0; dim < 3; ++dim)
  {
    const long int off = offset[dim];
    for (std::size_t n_iter = 0; n_iter < 2; ++n_iter)
    {
      std::swap (buffer, data);
     for(y = 1; y < small_height - 1; ++y)
      for(x = 1; x < small_width - 1; ++x)
        {
          Eigen::Vector2f* d_ptr = &(data (x,y,1));
          Eigen::Vector2f* b_ptr = &(buffer (x,y,1));

          for(std::size_t z = 1; z < small_depth - 1; ++z, ++d_ptr, ++b_ptr)
            *d_ptr = (*(b_ptr - off) + *(b_ptr + off) + 2.0 * (*b_ptr)) / 4.0;
        }
    }
  }

  if (early_division_)
  {
    for (std::vector<Eigen::Vector2f >::iterator d = data.begin (); d != data.end (); ++d)
      *d /= ((*d)[0] != 0) ? (*d)[1] : 1;
     
     for (y = 0; y < input_->height; y++)
      for (x = 0; x < input_->width; x++)
      {
        const float z = output (x,y).z - base_min;
        const Eigen::Vector2f D = data.trilinear_interpolation (static_cast<float> (x) / sigma_s_ + padding_xy,
                                                                static_cast<float> (y) / sigma_s_ + padding_xy,
                                                                z / sigma_r_ + padding_z);
        output(x,y).z = D[0];
      }
  }
  else
  {
    for (y = 0; y < input_->height; ++y)
      for (x = 0; x < input_->width; ++x)
      {
        const float z = output (x,y).z - base_min;
        const Eigen::Vector2f D = data.trilinear_interpolation (static_cast<float> (x) / sigma_s_ + padding_xy,
                                                                static_cast<float> (y) / sigma_s_ + padding_xy,
                                                                z / sigma_r_ + padding_z);
        output (x,y).z = D[0] / D[1];
      }
  }

bf->destroyInstance();  
}



//////////////////////////////////////////////////////////////////////////////////////////////
template <typename PointT> std::size_t
pcl::ocl::FastBilateralFilter<PointT>::Array3D::clamp (const std::size_t min_value,
                                                  const std::size_t max_value,
                                                  const std::size_t x)
{
  if (x >= min_value && x <= max_value)
  {
    return x;
  }
  else if (x < min_value)
  {
    return (min_value);
  }
  else
  {
    return (max_value);
  }
}

//////////////////////////////////////////////////////////////////////////////////////////////
template <typename PointT> Eigen::Vector2f
pcl::ocl::FastBilateralFilter<PointT>::Array3D::trilinear_interpolation (const float x,
                                                                    const float y,
                                                                    const float z)
{
  const std::size_t x_index  = clamp (0, x_dim_ - 1, static_cast<std::size_t> (x));
  const std::size_t xx_index = clamp (0, x_dim_ - 1, x_index + 1);

  const std::size_t y_index  = clamp (0, y_dim_ - 1, static_cast<std::size_t> (y));
  const std::size_t yy_index = clamp (0, y_dim_ - 1, y_index + 1);

  const std::size_t z_index  = clamp (0, z_dim_ - 1, static_cast<std::size_t> (z));
  const std::size_t zz_index = clamp (0, z_dim_ - 1, z_index + 1);

  const float x_alpha = x - static_cast<float> (x_index);
  const float y_alpha = y - static_cast<float> (y_index);
  const float z_alpha = z - static_cast<float> (z_index);

  return
      (1.0f-x_alpha) * (1.0f-y_alpha) * (1.0f-z_alpha) * (*this)(x_index, y_index, z_index) +
      x_alpha        * (1.0f-y_alpha) * (1.0f-z_alpha) * (*this)(xx_index, y_index, z_index) +
      (1.0f-x_alpha) * y_alpha        * (1.0f-z_alpha) * (*this)(x_index, yy_index, z_index) +
      x_alpha        * y_alpha        * (1.0f-z_alpha) * (*this)(xx_index, yy_index, z_index) +
      (1.0f-x_alpha) * (1.0f-y_alpha) * z_alpha        * (*this)(x_index, y_index, zz_index) +
      x_alpha        * (1.0f-y_alpha) * z_alpha        * (*this)(xx_index, y_index, zz_index) +
      (1.0f-x_alpha) * y_alpha        * z_alpha        * (*this)(x_index, yy_index, zz_index) +
      x_alpha        * y_alpha        * z_alpha        * (*this)(xx_index, yy_index, zz_index);
}

#endif /* PCL_FILTERS_IMPL_FAST_BILATERAL_HPP_ */
