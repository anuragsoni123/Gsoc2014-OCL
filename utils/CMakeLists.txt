set(SUBSYS_NAME ocl_utils)
set(SUBSYS_PATH ocl/utils)
set(SUBSYS_DESC "Device layer functions.")
set(SUBSYS_DEPS common)

PCL_SUBSYS_OPTION(build ${SUBSYS_NAME} ${SUBSYS_DESC} ON)
#mark_as_advanced("BUILD_${SUBSYS_NAME}")
PCL_SUBSYS_DEPEND(build ${SUBSYS_NAME} DEPS ${SUBSYS_DEPS})

if(BUILD_OCL)       
	set(srcs
        src/ocl_manager.cpp
        )
	set(incs
        include/pcl/ocl/utils/ocl_manager.h
        )
	set(impl_incs
        include/pcl/ocl/utils/impl/ocl_kernel_builder.hpp
	)
       
    set(LIB_NAME pcl_${SUBSYS_NAME})
    include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include "/usr/local/include/pcl-1.7")
    include_directories(${OPENCL_INCLUDE_DIRS})
    PCL_ADD_LIBRARY(${LIB_NAME} ${SUBSYS_NAME} ${srcs} ${incs} ${impl_incs})
    target_link_libraries(${LIB_NAME} pcl_common ${OPENCL_LIBRARIES})
    PCL_MAKE_PKGCONFIG(${LIB_NAME} ${SUBSYS_NAME} "${SUBSYS_DESC}" "${SUBSYS_DEPS}" "" "" "" "")

    # Install include files
    PCL_ADD_INCLUDES(${SUBSYS_NAME} ${SUBSYS_PATH} ${incs})
    PCL_ADD_INCLUDES(${SUBSYS_NAME} ${SUBSYS_PATH}/impl ${impl_incs})
endif(BUILD_OCL)
