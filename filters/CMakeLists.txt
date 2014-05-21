set(SUBSYS_NAME ocl_filters)
set(SUBSYS_PATH ocl/filters)
set(SUBSYS_DESC "Filters OCL")
set(SUBSYS_DEPS common filters sample_consensus search kdtree octree ocl_utils)

PCL_SUBSYS_OPTION(build ${SUBSYS_NAME} ${SUBSYS_DESC} ON)
mark_as_advanced("BUILD_${SUBSYS_NAME}")
PCL_SUBSYS_DEPEND(build ${SUBSYS_NAME} DEPS ${SUBSYS_DEPS})
PCL_SET_SUBSYS_INCLUDE_DIR(${SUBSYS_NAME} ${SUBSYS_PATH})

if(BUILD_OCL) 
	set(srcs
	src/VectorAddition.cpp
        src/median_filter.cpp
        #src/passthrough.cpp
        #src/statisticalOutlierRemoval.cpp
        #src/voxelgrid.cpp
        )
	set(cls
        include/pcl/ocl/filters/impl/cl/vector_add_kernel.cl
        include/pcl/ocl/filters/impl/cl/median_filter.cl
        #include/pcl/ocl/filters/impl/cl/passthrough.cl
        #include/pcl/ocl/filters/impl/cl/statistical_outlier_removal.cl
        #include/pcl/ocl/filters/impl/cl/min_max_3d.cl
        #include/pcl/ocl/filters/impl/cl/vxl_first_pass.cl
        #include/pcl/ocl/filters/impl/cl/total_reduction.cl
        )
    set(clbins
        include/pcl/ocl/filters/impl/cl/median_filter.clbin
        #include/pcl/ocl/filters/impl/cl/passthrough.cl
        #include/pcl/ocl/filters/impl/cl/statistical_outlier_removal.clbin
        #include/pcl/ocl/filters/impl/cl/min_max_3d.clbin
        #include/pcl/ocl/filters/impl/cl/vxl_first_pass.clbin
        #include/pcl/ocl/filters/impl/cl/total_reduction.clbin
        #include/pcl/ocl/filters/impl/cl/bitonic_sorting.clbin
        )
	set(incs
         include/pcl/ocl/filters/VectorAddition.h
         include/pcl/ocl/filters/median_filter.h
        #include/pcl/ocl/filters/passthrough.h
        #include/pcl/ocl/filters/statistical_outlier_removal.h
        #include/pcl/ocl/filters/voxel_grid.h
        )
        
	set(impl_incs
        include/pcl/ocl/filters/impl/median_filter.hpp
        #include/pcl/ocl/filters/impl/passthrough.hpp
        #include/pcl/ocl/filters/impl/statistical_outlier_removal.hpp
        #include/pcl/ocl/filters/impl/voxel_grid.hpp
        )
        
    set(LIB_NAME pcl_${SUBSYS_NAME})
    include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include ${OPENCL_INCLUDE_DIRS} "/usr/local/include/pcl-1.7")
    PCL_ADD_LIBRARY(${LIB_NAME} ${SUBSYS_NAME} ${srcs} ${incs} ${impl_incs})
    target_link_libraries(${LIB_NAME} pcl_common pcl_filters pcl_sample_consensus pcl_search pcl_kdtree pcl_octree pcl_ocl_utils ${OPENCL_LIBRARIES})
    PCL_MAKE_PKGCONFIG(${LIB_NAME} ${SUBSYS_NAME} "${SUBSYS_DESC}" "${SUBSYS_DEPS}" "" "" "" "")

    # Install include files
    PCL_ADD_INCLUDES(${SUBSYS_NAME} ${SUBSYS_PATH} ${incs})
    PCL_ADD_INCLUDES(${SUBSYS_NAME} ${SUBSYS_PATH}/impl ${impl_incs})
    PCL_ADD_INCLUDES(${SUBSYS_NAME} ${SUBSYS_PATH}/impl/cl ${cls})
    PCL_ADD_INCLUDES(${SUBSYS_NAME} ${SUBSYS_PATH}/impl/clbin ${clbins})
 
endif(BUILD_OCL)
