cmake_minimum_required (VERSION 3.7.2 FATAL_ERROR)


if(WIN32)
project (msk144_cu_decoder
  VERSION 1.0.0.0
  LANGUAGES C CXX CUDA
  )
else()
project (msk144_cu_decoder
  VERSION 1.0.0.0
  LANGUAGES C CXX CUDA Fortran 
  )
endif()

set(CMAKE_CXX_STANDARD 17)

set(WSJT_BASE_DIR ${CMAKE_SOURCE_DIR}/deps/wsjtx CACHE STRING "Provide path to WSJT sorurces")

if(WIN32)
add_definitions (-D_CRT_SECURE_NO_WARNINGS=1)
endif()

# set CMAKE_BUILD_TYPE to Release if not defined
if (NOT CMAKE_BUILD_TYPE)
  set (CMAKE_BUILD_TYPE RELEASE CACHE STRING
    "Choose the type of build, options are: None Debug Release."
    FORCE)
endif (NOT CMAKE_BUILD_TYPE)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 35 53 61)
endif()

if(WIN32)
  include_directories(${CMAKE_SOURCE_DIR}/ext/getopt-for-windows)
endif()

include_directories(src)

# project definitions
add_definitions (-DBIGSYM=1)

#add_definitions (-DUSE_SIMPLE_METRICS=1)
#add_definitions (-DUSE_SIMPLE_GPU_TIMER=1)


if(WIN32)
  add_definitions (-DUSE_EXTERNAL_PBDECODE=1)
else()
  set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -pthread")
  set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --std=gnu++11 -pthread")

  set (General_FFLAGS "-Wall -Wno-conversion -fno-second-underscore")
  
  # gfortran specific settings
  set (CMAKE_Fortran_FLAGS_RELEASE "${CMAKE_Fortran_FLAGS_RELEASE} -fbounds-check -funroll-all-loops -fno-f2c ${General_FFLAGS}")
  set (CMAKE_Fortran_FLAGS_DEBUG   "${CMAKE_Fortran_FLAGS_DEBUG} -fbounds-check -fno-f2c ${General_FFLAGS}")
endif()


if(WIN32)
  set(EXTRA_SRC_FILES ext/getopt-for-windows/getopt.c )
endif()

set (wsjt_common_FSRCS
  ${WSJT_BASE_DIR}/lib/packjt.f90
  ${WSJT_BASE_DIR}/lib/deg2grid.f90
  ${WSJT_BASE_DIR}/lib/grid2deg.f90
  ${WSJT_BASE_DIR}/lib/fmtmsg.f90
  )

set (wsjt_msk144_FSRCS
  ${WSJT_BASE_DIR}/lib/77bit/packjt77.f90
  ${WSJT_BASE_DIR}/lib/chkcall.f90
  )


set (wsjt_msk144_CSRCS
  )


if(NOT WIN32)  
  set(PBDECODE_FILES   
    ${wsjt_msk144_CSRCS}  
    ${wsjt_common_FSRCS} 
    ${wsjt_msk144_FSRCS} 
    )
endif()

add_executable (msk144cudecoder 
  src/common.h
  src/gpu_timer.h
  src/msk_context.cuh
  src/result_keeper.cuh
  src/sum_reduction.cuh
  src/scan_kernel.cuh 
  src/softbits_kernel.cuh 
  src/main.cu
  src/analytic_fft.h
  src/analytic_fft.cu
  src/ldpc_kernel.cuh
  src/ldpc_context.cuh
  src/index_kernel.cuh
  src/analytic2.cuh
  src/decode_softbits.h
  src/decode_softbits.cpp
  src/metrics.h
  src/metrics.cpp
  src/snr_tracker.h
  src/snr_tracker.cu
  src/smath_complex.h
  src/pattern_item.h
  src/f_interop.h
  src/f_interop.cpp
  src/result_filter.h
  src/result_filter.cpp
  ${EXTRA_SRC_FILES}
  ${PBDECODE_FILES}
  )


if(WIN32)
  target_link_libraries(msk144cudecoder "cufft.lib")
else()
  target_link_libraries(msk144cudecoder "/usr/local/cuda/lib64/libcufft.so")
endif()


if(USE_FAST_MATH)
  target_compile_options(msk144cudecoder PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>) 
endif()

