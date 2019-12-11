cmake_minimum_required(VERSION 3.5.1)
project(meshflow)

set(CMAKE_CXX_STANDARD 11)

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug)
endif()

message("Build type: " ${CMAKE_BUILD_TYPE})
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall") # -O3 -march=native
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -fPIC -g") # -O3 -march=native

LIST(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules)

find_package(OpenCV REQUIRED)
if (NOT OpenCV_FOUND)
    message(FATAL_ERROR "OpenCV not found.")
endif()

find_package(Lemon REQUIRED)
if(NOT LEMON_FOUND)
        MESSAGE(FATAL_ERROR "Lemon not found.")
endif()

find_package(Clp REQUIRED)
if(NOT Clp_FOUND)
        MESSAGE(FATAL_ERROR "Clp not found.")
endif()

message("Lemon incl: ${LEMON_INCLUDE_DIR}")
message("Lemon libs: ${LEMON_LIBRARIES}")
message("Clp incl: ${Clp_INCLUDE_DIRS}")
message("Clp libs: ${Clp_LIBRARIES}")

include_directories(
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/include
    #${LEMON_INCLUDE_DIR}
)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)

add_library(${PROJECT_NAME} SHARED
    src/System.cpp
    src/OpticalFlow.cpp
    src/MeshFlow.cpp
    src/Optimization.cpp
    src/Configurations.cpp
    src/evaluation.cpp
)

target_link_libraries(${PROJECT_NAME}
    ${OpenCV_LIBS}
    ${Clp_LIBRARIES}
    ${LEMON_LIBRARIES}
)

# Build example binaries
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/examples)

add_executable(stabilize
    examples/stabilize.cpp
)
target_link_libraries(stabilize ${PROJECT_NAME})

