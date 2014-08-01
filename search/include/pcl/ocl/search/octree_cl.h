/*
 * Software License Agreement (BSD License)
 *
 *  Point Cloud Library (PCL) - www.pointclouds.org
 *  Copyright (c) 2010-2011, Willow Garage, Inc.
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
 */

#ifndef PCL_SEARCH_OCTREECL_CL_H
#define PCL_SEARCH_OCTREECL_CL_H

#include <pcl/ocl/search/search.h>
#include <pcl/ocl/octree/impl/octree.hpp>

namespace pcl
{
namespace ocl
 {
  namespace search
  {
    /** \brief @b search::Octree is a wrapper class which implements nearest neighbor search operations based on the 
      * pcl::octree::Octree structure. 
      * 
      * resolution. Its bounding box is automatically adjusted according to the
      * pointcloud dimension or it can be predefined. Note: The tree depth
      * equates to the resolution and the bounding box dimensions of the
      * octree.
      * The octree pointcloud class needs to be initialized with its voxel
      */
    template<typename PointT>    
    class OctreeCL: public Search<PointT>
    {
      public:
        // public typedefs
        
        typedef boost::shared_ptr<std::vector<int> > IndicesPtr;
        typedef boost::shared_ptr<const std::vector<int> > IndicesConstPtr;

        typedef pcl::PointCloud<PointT> PointCloud;
        typedef boost::shared_ptr<PointCloud> PointCloudPtr;
        typedef boost::shared_ptr<const PointCloud> PointCloudConstPtr;

        
        typedef pcl::ocl::Octree OctreePointCloudSearchPtr;
        //typedef const pcl::ocl::octree::OctreePointCloudSearch<PointT> OctreePointCloudSearchConstPtr;
        OctreePointCloudSearchPtr tree_;

        using pcl::ocl::search::Search<PointT>::input_;
        using pcl::ocl::search::Search<PointT>::indices_;
        using pcl::ocl::search::Search<PointT>::sorted_results_;
        
        /** \brief Octree constructor.
          */
        OctreeCL()
          : Search<PointT> ("OctreeCL")
        {
        }

        /** \brief Empty Destructor. */
        virtual
        ~OctreeCL ()
        {
        }

        /** \brief Provide a pointer to the input dataset.
          * \param[in] cloud the const boost shared pointer to a PointCloud message
          */
        inline void
        setInputCloud (const PointCloudConstPtr &cloud)
        {
           //tree.clear();
           tree_.setCloud(*cloud);
           input_ = cloud;
          /*tree_->deleteTree ();
          tree_->setInputCloud (cloud);
          tree_->addPointsFromInputCloud();
          input_ = cloud;*/
        }
        
        /** \brief search for all neighbors of query point that are within a given radius.
         * \param p_q the given query point
         * \param radius the radius of the sphere bounding all of p_q's neighbors
         * \param k_indices the resultant indices of the neighboring points
         * \param k_sqr_distances the resultant squared distances to the neighboring points
         * \param max_nn if given, bounds the maximum returned neighbors to this value
         * \return number of neighbors found in radius
         */
        inline int
        radiusSearch (const PointT &p_q, 
                      float radius, 
                      std::vector<int> &k_indices,
                      std::vector<float> &k_sqr_distances, 
                      unsigned int max_nn = 0) const
        {
          tree_.radiusSearch (p_q, radius, k_indices, k_sqr_distances, max_nn);
          //if (sorted_results_)
            //this->sortResults (k_indices, k_sqr_distances);
          //return (static_cast<int> (k_indices.size ()));
          return 0;
        }

        
        /** \brief Search for approximate nearest neighbor at the query point.
          * \param[in] p_q the given query point
          * \param[out] result_index the resultant index of the neighbor point
          * \param[out] sqr_distance the resultant squared distance to the neighboring point
          */
        inline void
        approxNearestSearch (const PointT &p_q, int &result_index, float &sqr_distance)
        {
          tree_.approxNearestSearch (p_q, result_index, sqr_distance);
        }

    };
 
  }
 }
}

#ifdef PCL_NO_PRECOMPILE
//#include <pcl/ocl/octree/impl/octree_search.hpp>
#else
#define PCL_INSTANTIATE_OctreeCL(T) template class PCL_EXPORTS pcl::ocl::search::OctreeCL<T>;
#endif

#endif    // PCL_SEARCH_OCTREE_H
