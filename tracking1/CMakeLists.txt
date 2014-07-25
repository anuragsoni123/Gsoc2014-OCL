set(SUBSYS_NAME ocl_tracking)
set(SUBSYS_PATH ocl/tracking)
set(SUBSYS_DESC "Point cloud tracking library with OpenCL")
set(SUBSYS_DEPS common search kdtree filters octree ocl_utils)

PCL_SUBSYS_OPTION(build ${SUBSYS_NAME} ${SUBSYS_DESC} ON)
mark_as_advanced("BUILD_${SUBSYS_NAME}")
PCL_SUBSYS_DEPEND(build ${SUBSYS_NAME} DEPS ${SUBSYS_DEPS})
PCL_SET_SUBSYS_INCLUDE_DIR(${SUBSYS_NAME} ${SUBSYS_PATH})


if(BUILD_OCL)
    set(srcs 
        src/tracking.cpp
        src/particle_filter.cpp
        src/kld_adaptive_particle_filter.cpp
        src/coherence.cpp
        )

    set(incs 
        "include/pcl/ocl/tracking/boost.h"
        "include/pcl/ocl/tracking/tracking.h"
        "include/pcl/ocl/tracking/tracker.h"
        "include/pcl/ocl/tracking/coherence.h"
        "include/pcl/ocl/tracking/nearest_pair_point_cloud_coherence.h"
        "include/pcl/ocl/tracking/approx_nearest_pair_point_cloud_coherence.h"
        "include/pcl/ocl/tracking/distance_coherence.h"
        "include/pcl/ocl/tracking/hsv_color_coherence.h"
        "include/pcl/ocl/tracking/normal_coherence.h"
        "include/pcl/ocl/tracking/particle_filter.h"
        "include/pcl/ocl/tracking/particle_filter_omp.h"
        "include/pcl/ocl/tracking/kld_adaptive_particle_filter.h"
        "include/pcl/ocl/tracking/kld_adaptive_particle_filter_omp.h"
        "include/pcl/ocl/tracking/pyramidal_klt.h"
        )

    set(impl_incs 
        "include/pcl/ocl/tracking/impl/tracking.hpp"
        "include/pcl/ocl/tracking/impl/tracker.hpp"
        "include/pcl/ocl/tracking/impl/coherence.hpp"
        "include/pcl/ocl/tracking/impl/nearest_pair_point_cloud_coherence.hpp"
        "include/pcl/ocl/tracking/impl/approx_nearest_pair_point_cloud_coherence.hpp"
        "include/pcl/ocl/tracking/impl/distance_coherence.hpp"
        "include/pcl/ocl/tracking/impl/hsv_color_coherence.hpp"
        "include/pcl/ocl/tracking/impl/normal_coherence.hpp"
        "include/pcl/ocl/tracking/impl/particle_filter.hpp"
        "include/pcl/ocl/tracking/impl/particle_filter_omp.hpp"
        "include/pcl/ocl/tracking/impl/kld_adaptive_particle_filter.hpp"
        "include/pcl/ocl/tracking/impl/kld_adaptive_particle_filter_omp.hpp"
        "include/pcl/ocl/tracking/impl/pyramidal_klt.hpp"
	"include/pcl/ocl/tracking/impl/min_max.hpp"
        )

    set(LIB_NAME "pcl_${SUBSYS_NAME}")
    include_directories("${CMAKE_CURRENT_SOURCE_DIR}/include" ${OPENCL_INCLUDE_DIRS} "/usr/local/include/pcl-1.7")
    PCL_ADD_LIBRARY("${LIB_NAME}" "${SUBSYS_NAME}" ${srcs} ${incs} ${impl_incs})
    target_link_libraries("${LIB_NAME}" pcl_common pcl_kdtree pcl_search pcl_filters pcl_octree pcl_ocl_utils ${OPENCL_LIBRARIES})
    PCL_MAKE_PKGCONFIG("${LIB_NAME}" "${SUBSYS_NAME}" "${SUBSYS_DESC}" "${SUBSYS_DEPS}" "" "" "" "")

    # Install include files
    PCL_ADD_INCLUDES("${SUBSYS_NAME}" "${SUBSYS_PATH}" ${incs})
    PCL_ADD_INCLUDES("${SUBSYS_NAME}" "${SUBSYS_PATH}/impl" ${impl_incs})
endif(BUILD_OCL)
