set(SUBSYS_NAME ocl_common)
set(SUBSYS_PATH ocl/common)
set(SUBSYS_DESC "Common OCL")
set(SUBSYS_DEPS common sample_consensus search kdtree octree ocl_utils)

PCL_SUBSYS_OPTION(build ${SUBSYS_NAME} ${SUBSYS_DESC} OFF)
mark_as_advanced("BUILD_${SUBSYS_NAME}")
PCL_SUBSYS_DEPEND(build ${SUBSYS_NAME} DEPS ${SUBSYS_DEPS})
PCL_SET_SUBSYS_INCLUDE_DIR(${SUBSYS_NAME} ${SUBSYS_PATH})

if(BUILD_OCL)
	set(srcs
        src/centroid.cpp
        src/common.cpp
        src/cpa.cpp
	src/gaussian.cpp
        src/polynomial_calculations.cpp
        src/projection_matrix.cpp
        src/spring.cpp
        src/transforms.cpp
        )
	set(cls
        src/cl/centroid.cl
        src/cl/common.cl
        src/cl/cpa.cl
	src/cl/gaussian.cl
        src/cl/polynomial_calculations.cl
        src/cl/projection_matrix.cl
        src/cl/spring.cl
        src/cl/transforms.cl
        )
	set(incs
        include/pcl/ocl/common/centroid.h
        include/pcl/ocl/common/common.h
        include/pcl/ocl/common/cpa.h
        include/pcl/ocl/common/gaussian.h
        include/pcl/ocl/common/polynomial_calculations.h
        include/pcl/ocl/common/projection_matrix.h
        include/pcl/ocl/common/spring.h
        include/pcl/ocl/common/transforms.h
        )
    set(LIB_NAME pcl_${SUBSYS_NAME})
    include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include ${OPENCL_INCLUDE_DIRS})
    PCL_ADD_LIBRARY(${LIB_NAME} ${SUBSYS_NAME} ${srcs} ${incs} ${impl_incs})
    target_link_libraries(${LIB_NAME} pcl_common pcl_sample_consensus pcl_search pcl_kdtree pcl_octree pcl_ocl_utils ${OPENCL_LIBRARIES})
    PCL_MAKE_PKGCONFIG(${LIB_NAME} ${SUBSYS_NAME} "${SUBSYS_DESC}" "${SUBSYS_DEPS}" "" "" "" "")

    # Install include files
    PCL_ADD_INCLUDES(${SUBSYS_NAME} ${SUBSYS_PATH} ${incs})
    PCL_ADD_INCLUDES(${SUBSYS_NAME} ${SUBSYS_PATH}/impl ${impl_incs})
    PCL_ADD_INCLUDES(${SUBSYS_NAME} ${SUBSYS_PATH}/impl/cl ${cls})
endif(BUILD_OCL)
