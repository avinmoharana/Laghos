cmake_minimum_required(VERSION 2.8.11)
project(laghos)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/config)

option(ENABLE_SIMMETRIX "Build with Simmetrix support" OFF)
if(ENABLE_SIMMETRIX)
  add_definitions(-DHAVE_SIMMETRIX)
endif()

if(ENABLE_SIMMETRIX)
  find_package(SimModSuite REQUIRED)
endif()

find_package(Threads REQUIRED)
find_package(HYPRE REQUIRED)
find_package(METIS REQUIRED)

if (SCOREC_PREFIX)
  find_package(SCOREC 2.1.0 REQUIRED CONFIG PATHS ${SCOREC_PREFIX} NO_DEFAULT_PATH)
else()
# IF SCOREC_PREFIX was not specified, look in typical system directories,
# and also in CMAKE_PREFIX_PATH (environment variable)
  find_package(
      SCOREC #package name, has to be SCOREC
      2.1.0  #version. can be omitted, and will match any installed version
             #greater than or equal to this one, as long as the major number
             #is the same
      REQUIRED #indicate that SCOREC is really needed to compile
      CONFIG   #skip the 'MODULE' search system, save some time and confusion
      )
endif()

if(ENABLE_SIMMETRIX)
  message("SIMMODSUITE_LIBS is ${SIMMODSUITE_LIBS}")
endif()

add_library(laghos_assembly STATIC ${CMAKE_CURRENT_SOURCE_DIR}/laghos_assembly.cpp)
include_directories(${MFEM_INCLUDE_DIR})
include_directories(${PUMI_INCLUDE_DIR})
target_link_libraries(laghos_assembly ${CMAKE_THREAD_LIBS_INIT} ${MFEM_LIB_DIR}/libmfem.a)

add_library(laghos_solver STATIC ${CMAKE_CURRENT_SOURCE_DIR}/laghos_solver.cpp)
include_directories(${MFEM_INCLUDE_DIR})
include_directories(${PUMI_INCLUDE_DIR})
target_link_libraries(laghos_solver ${CMAKE_THREAD_LIBS_INIT} ${MFEM_LIB_DIR}/libmfem.a)


add_executable(laghos ${CMAKE_CURRENT_SOURCE_DIR}/laghos.cpp)
include_directories(${SIMMODSUITE_INCLUDE_DIR})
include_directories(${MFEM_INCLUDE_DIR})
include_directories(${PUMI_INCLUDE_DIR})

target_link_libraries(laghos ${MFEM_LIB_DIR}/libmfem.a
                             ${HYPRE_LIBRARIES}
                             ${METIS_LIBRARIES}
                             SCOREC::core
                             laghos_solver
                             laghos_assembly)
if(ENABLE_SIMMETRIX)
  target_link_libraries(laghos ${SIMMODSUITE_LIBS})
endif()
