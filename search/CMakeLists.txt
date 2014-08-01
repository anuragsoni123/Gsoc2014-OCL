set(SUBSYS_NAME ocl_search)
set(SUBSYS_PATH ocl/search)
set(SUBSYS_DESC "Search OCL")
set(SUBSYS_DEPS common search kdtree filters octree io ocl_utils)

PCL_SUBSYS_OPTION(build ${SUBSYS_NAME} ${SUBSYS_DESC} ON)
mark_as_advanced("BUILD_${SUBSYS_NAME}")
PCL_SUBSYS_DEPEND(build ${SUBSYS_NAME} DEPS ${SUBSYS_DEPS})
PCL_SET_SUBSYS_INCLUDE_DIR(${SUBSYS_NAME} ${SUBSYS_PATH})

if(BUILD_OCL) 
	set(srcs
        src/brute_force.cpp
        src/kdtree.cpp
        src/octree.cpp
	src/organized.cpp
        src/search.cpp
	src/octree_cl.cpp
        )
	#set(cls
        #src/cl/brute_force.cl
        #src/cl/kdtree.cl
        #src/cl/octree.cl
	#src/cl/organized.cl
        #)
	set(incs
        include/pcl/ocl/search/brute_force.h
        include/pcl/ocl/search/kdtree.h
        include/pcl/ocl/search/octree.h
        include/pcl/ocl/search/organized.h
        include/pcl/ocl/search/search.h
	include/pcl/ocl/search/octree_cl.h
        )
    set(LIB_NAME pcl_${SUBSYS_NAME})
    include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include ${OPENCL_INCLUDE_DIRS} "/usr/local/include/pcl-1.7")
    PCL_ADD_LIBRARY(${LIB_NAME} ${SUBSYS_NAME} ${srcs} ${incs} ${impl_incs})
    target_link_libraries(${LIB_NAME} pcl_common pcl_io pcl_ocl_utils pcl_common pcl_kdtree pcl_search pcl_filters pcl_octree ${OPENCL_LIBRARIES})
    PCL_MAKE_PKGCONFIG(${LIB_NAME} ${SUBSYS_NAME} "${SUBSYS_DESC}" "${SUBSYS_DEPS}" "" "" "" "")

    # Install include files
    PCL_ADD_INCLUDES(${SUBSYS_NAME} ${SUBSYS_PATH} ${incs})
    PCL_ADD_INCLUDES(${SUBSYS_NAME} ${SUBSYS_PATH}/impl ${impl_incs})
    #PCL_ADD_INCLUDES(${SUBSYS_NAME} ${SUBSYS_PATH}/impl/cl ${cls})
endif(BUILD_OCL)
