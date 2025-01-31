cmake_minimum_required(VERSION 3.12.0)
project(project LANGUAGES CXX C)

include(../common/cmake-common.cmake)

add_executable(project src/project.cpp src/WaveSolver.cpp)
deal_ii_setup_target(project)