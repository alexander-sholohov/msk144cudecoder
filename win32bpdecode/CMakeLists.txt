cmake_minimum_required (VERSION 3.7.2)


project (bpdecode
  VERSION 1.0.0.0
  LANGUAGES C CXX Fortran
  )

# set CMAKE_BUILD_TYPE to Release if not defined
if (NOT CMAKE_BUILD_TYPE)
  set (CMAKE_BUILD_TYPE RELEASE CACHE STRING
    "Choose the type of build, options are: None Debug Release."
    FORCE)
endif (NOT CMAKE_BUILD_TYPE)

set(WSJT_BASE_DIR ${CMAKE_SOURCE_DIR}/../deps/wsjtx CACHE STRING "Provide path to WSJT sorurces")

# Hint: Define BOOST_ROOT if boost is not in standart path
find_package (Boost REQUIRED)
include_directories(${Boost_INCLUDE_DIR})

# project definitions
add_definitions (-DBIGSYM=1)

set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -pthread")
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --std=gnu++11 -pthread ")
#set (CMAKE_CXX_FLAGS_INIT "")

set (General_FFLAGS "-Wall -Wno-conversion -fno-second-underscore")

# gfortran specific settings
set (CMAKE_Fortran_FLAGS_RELEASE "${CMAKE_Fortran_FLAGS_RELEASE} -fbounds-check -funroll-all-loops -fno-f2c ${General_FFLAGS}")
set (CMAKE_Fortran_FLAGS_DEBUG   "${CMAKE_Fortran_FLAGS_DEBUG} -fbounds-check -fno-f2c ${General_FFLAGS}")


set (wsjt_common_FSRCS
  ${WSJT_BASE_DIR}/lib/packjt.f90
  ${WSJT_BASE_DIR}/lib/deg2grid.f90
  ${WSJT_BASE_DIR}/lib/grid2deg.f90
  ${WSJT_BASE_DIR}/lib/fmtmsg.f90
  )

set (wsjt_msk144_FSRCS
  ${WSJT_BASE_DIR}/lib/crc.f90
  ${WSJT_BASE_DIR}/lib/77bit/packjt77.f90
  ${WSJT_BASE_DIR}/lib/bpdecode128_90.f90
  ${WSJT_BASE_DIR}/lib/ft8/chkcrc13a.f90
  ${WSJT_BASE_DIR}/lib/chkcall.f90
  ${WSJT_BASE_DIR}/lib/platanh.f90
  )

set (wsjt_CXXSRCS
  ${WSJT_BASE_DIR}/lib/crc13.cpp
  )

set (wsjt_msk144_CSRCS
  )

add_library(bpdecode SHARED
  ${wsjt_CXXSRCS} 
  ${wsjt_msk144_CSRCS}  
  ${wsjt_common_FSRCS} 
  ${wsjt_msk144_FSRCS} 
  )

target_link_libraries(bpdecode stdc++)

#copy result to dll file
add_custom_command(TARGET bpdecode 
                   POST_BUILD
                   COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:bpdecode> bpdecode.dll)
