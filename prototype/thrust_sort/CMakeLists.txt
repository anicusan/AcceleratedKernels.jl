# Base project details
cmake_minimum_required(VERSION 3.23)

project(
    BaskervilleUtilitiesCollection
    VERSION 0.1
    LANGUAGES CXX C CUDA)

# Find Thrust and create a "parent" target to link our libraries to
find_package(Thrust REQUIRED CONFIG)
thrust_create_target(Thrust DEVICE CUDA)

# Create our library and link it to the parent Thrust target
add_library(BUCLib SHARED buc.hpp buc.cu)
target_include_directories(BUCLib PUBLIC .)
target_link_libraries(BUCLib Thrust)

# Compile CUDA code only for native architecture
set_target_properties(
    Thrust BUCLib
    PROPERTIES CUDA_ARCHITECTURES native)

# Create example test executables
add_subdirectory(tests)

# Installing / saving paths
install(
    TARGETS BUCLib
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin)

