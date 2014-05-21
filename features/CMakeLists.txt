set(SUBSYS_NAME ocl_features)
set(SUBSYS_PATH ocl/features)
set(SUBSYS_DESC "Features OCL")
set(SUBSYS_DEPS common io search ocl_utils)

PCL_SUBSYS_OPTION(build ${SUBSYS_NAME} ${SUBSYS_DESC} OFF)
mark_as_advanced("BUILD_${SUBSYS_NAME}")
PCL_SUBSYS_DEPEND(build ${SUBSYS_NAME} DEPS ${SUBSYS_DEPS})
PCL_SET_SUBSYS_INCLUDE_DIR(${SUBSYS_NAME} ${SUBSYS_PATH})

if(BUILD_OCL)
	set(srcs
        src/features.cpp
        )
	set(cls
        src/cl/normal_3d.cl
        src/cl/pfh.cl
        src/cl/ppf.cl
        src/cl/principal_curvatures.cl
        src/cl/spinimages.cl
        src/cl/uniq_inds.cl
        src/cl/vfh.cl
        )
	set(incs
        include/pcl/ocl/features/features.h
        )
    set(LIB_NAME pcl_${SUBSYS_NAME})
    include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include ${OPENCL_INCLUDE_DIRS})
    PCL_ADD_LIBRARY(${LIB_NAME} ${SUBSYS_NAME} ${srcs} ${incs} ${impl_incs})
    target_link_libraries(${LIB_NAME} pcl_common pcl_io pcl_search pcl_kdtree pcl_octree pcl_ocl_utils ${OPENCL_LIBRARIES})
    PCL_MAKE_PKGCONFIG(${LIB_NAME} ${SUBSYS_NAME} "${SUBSYS_DESC}" "${SUBSYS_DEPS}" "" "" "" "")

    # Install include files
    PCL_ADD_INCLUDES(${SUBSYS_NAME} ${SUBSYS_PATH} ${incs})
    PCL_ADD_INCLUDES(${SUBSYS_NAME} ${SUBSYS_PATH}/impl ${impl_incs})
    PCL_ADD_INCLUDES(${SUBSYS_NAME} ${SUBSYS_PATH}/impl/cl ${cls})
endif(BUILD_OCL)
