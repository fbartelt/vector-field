cmake_minimum_required(VERSION 3.14)
project(pylie)

# Find Pybind11
find_package(pybind11 REQUIRED)

include_directories(/usr/include/eigen3)

# Add the library
pybind11_add_module(pylie binder.cpp SpecialEuclideanGroup.hpp SpecialEuclideanAlgebra.hpp GeneralLinearGroup.hpp GeneralLinearAlgebra.hpp)
