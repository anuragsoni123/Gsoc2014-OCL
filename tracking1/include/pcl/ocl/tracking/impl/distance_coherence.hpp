#ifndef PCL_TRACKING_IMPL_DISTANCE_COHERENCE_CL_H_
#define PCL_TRACKING_IMPL_DISTANCE_COHERENCE_CL_H_

#include <Eigen/Dense>
#include <pcl/ocl/tracking/distance_coherence.h>

namespace pcl
{
 namespace ocl
 {
  namespace tracking
  {
    template <typename PointInT> double
    DistanceCoherence<PointInT>::computeCoherence (PointInT &source, PointInT &target)
    {
       Eigen::Vector4f p = source.getVector4fMap ();
       Eigen::Vector4f p_dash = target.getVector4fMap ();
       double d = (p - p_dash).norm ();
       return 1.0 / (1.0 + d * d * weight_);
    }
  }
 }
}

#define PCL_INSTANTIATE_DistanceCoherenceOCL(T) template class PCL_EXPORTS pcl::ocl::tracking::DistanceCoherence<T>;

#endif
