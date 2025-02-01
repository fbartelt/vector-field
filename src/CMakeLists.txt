cmake_minimum_required(VERSION 3.14)
project(_vectorfield)

# Find Pybind11
find_package(pybind11 REQUIRED)

# Find Python package for Python include directories
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)

# Optionally, find Eigen3 if it's required by your code
find_package(Eigen3 REQUIRED)

# Add the library (module) from source file binder.cpp
pybind11_add_module(_vectorfield binder.cpp)

# Add Eigen and Python include directories to the target
target_include_directories(_vectorfield PRIVATE ${EIGEN3_INCLUDE_DIR} ${Python3_INCLUDE_DIRS})

# Specify where the .so (shared object) will be placed
set_target_properties(_vectorfield PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/../vectorfield
    LIBRARY_OUTPUT_DIRECTORY_RELEASE ${CMAKE_SOURCE_DIR}/../vectorfield
    LIBRARY_OUTPUT_DIRECTORY_DEBUG ${CMAKE_SOURCE_DIR}/../vectorfield
)


# Ensure that the target is built correctly with the appropriate Python and Eigen dependencies
target_link_libraries(_vectorfield PRIVATE Python3::Python Eigen3::Eigen)
